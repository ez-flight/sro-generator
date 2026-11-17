"""Автоматические тесты генератора района СРО."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import shape

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.export import export_scenario
from sro_generator.generator import generate_scenario
from sro_generator.placement import PlacementError

ROOT = Path(__file__).resolve().parent.parent


def make_config(**overrides) -> ScenarioConfig:
    data = json.loads((ROOT / "example_config.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        parts = key.split("__")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    try:
        return ScenarioConfig.model_validate(data)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Ошибка в конфигурации JSON: {exc}") from exc


@pytest.fixture(scope="module")
def scenario():
    return generate_scenario(make_config())


class TestGeneration:
    def test_count_exactly_matches(self, scenario):
        assert len(scenario.objects) == scenario.config.objects.count

    def test_polygons_valid(self, scenario):
        for o in scenario.objects:
            assert o.geometry.is_valid
            assert o.geometry.is_simple
            assert 3 <= o.vertex_count <= 10
            assert len(o.geometry.exterior.coords) - 1 == o.vertex_count
            assert o.geometry.area > 0

    def test_areas_in_range(self, scenario):
        cfg = scenario.config.objects
        for o in scenario.objects:
            assert cfg.area_min_m2 * 0.99 <= o.area_m2 <= cfg.area_max_m2 * 1.01

    def test_objects_inside_district_with_margin(self, scenario):
        inner = scenario.district_local.buffer(-scenario.config.placement.boundary_margin_m)
        for o in scenario.objects:
            assert inner.covers(o.geometry)

    def test_min_distance_and_no_overlap(self, scenario):
        min_d = scenario.config.placement.min_distance_m
        objs = scenario.objects
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                assert a.geometry.distance(b.geometry) >= min_d - 1e-9
                assert a.geometry.intersection(b.geometry).area == 0.0

    def test_seed_reproducibility(self):
        s1 = generate_scenario(make_config())
        s2 = generate_scenario(make_config())
        for a, b in zip(s1.objects, s2.objects):
            assert a.object_id == b.object_id
            assert a.geometry.equals_exact(b.geometry, 1e-9)
            assert a.priority == b.priority
            assert a.deadline_h == b.deadline_h
            assert a.resolution_m == b.resolution_m
        assert [f.frame_id for f in s1.frames] == [f.frame_id for f in s2.frames]

    def test_different_seed_changes_result(self):
        s1 = generate_scenario(make_config())
        s2 = generate_scenario(make_config(seed=1234))
        coords1 = [o.geometry.exterior.coords[0] for o in s1.objects]
        coords2 = [o.geometry.exterior.coords[0] for o in s2.objects]
        assert coords1 != coords2

    def test_object_ids_stable(self, scenario):
        assert [o.object_id for o in scenario.objects] == [f"OBJ-{i:03d}" for i in range(1, 26)]

    def test_attributes_present(self, scenario):
        for o in scenario.objects:
            assert o.class_label in ("warehouse", "industrial", "transport")
            assert 1 <= o.priority <= 5
            assert o.deadline_h in (30.0, 48.0)
            assert o.resolution_m in (0.5, 1.0)
            assert o.placement_mode == "random"
            assert o.coordinate_fixed is False


class TestManualCenters:
    def test_manual_centers_kept(self):
        cfg = make_config(objects__count=5)
        from sro_generator.config import ManualCenter
        cfg.placement.manual_centers = [
            ManualCenter(object_id="OBJ-001", latitude=55.0, longitude=60.0)
        ]
        sc = generate_scenario(cfg)
        manual = sc.objects[0]
        assert manual.coordinate_fixed and manual.placement_mode == "manual"
        cx, cy = sc.local.to_xy(60.0, 55.0)
        assert abs(manual.center.x - cx) < 1e-6
        assert abs(manual.center.y - cy) < 1e-6
        assert len(sc.objects) == 5

    def test_manual_count_cannot_exceed_count(self):
        from sro_generator.config import ManualCenter
        with pytest.raises(ConfigError):
            make_config(
                objects__count=1,
                placement__manual_centers=[
                    ManualCenter(object_id="A", latitude=55.0, longitude=60.0).model_dump(),
                    ManualCenter(object_id="B", latitude=55.1, longitude=60.1).model_dump(),
                ],
            )


class TestValidation:
    def test_bad_vertices(self):
        with pytest.raises(ConfigError):
            make_config(objects__vertices_min=2)
        with pytest.raises(ConfigError):
            make_config(objects__vertices_max=11)
        with pytest.raises(ConfigError):
            make_config(objects__vertices_min=8, objects__vertices_max=4)

    def test_bad_areas(self):
        with pytest.raises(ConfigError):
            make_config(objects__area_min_m2=-1.0)
        with pytest.raises(ConfigError):
            make_config(objects__area_min_m2=10.0, objects__area_max_m2=5.0)

    def test_bad_district(self):
        with pytest.raises(ConfigError):
            make_config(district__center_lat=91.0)
        with pytest.raises(ConfigError):
            make_config(district__width_km=0.0)

    def test_bad_overlap(self):
        with pytest.raises(ConfigError):
            make_config(placement__max_overlap_fraction=1.5)
        with pytest.raises(ConfigError):
            make_config(placement__allow_overlap=False, placement__max_overlap_fraction=0.3)

    def test_duplicate_manual_ids(self):
        from sro_generator.config import ManualCenter
        with pytest.raises(ConfigError):
            make_config(
                placement__manual_centers=[
                    ManualCenter(object_id="X", latitude=55.0, longitude=60.0).model_dump(),
                    ManualCenter(object_id="X", latitude=55.2, longitude=60.2).model_dump(),
                ]
            )


class TestPlacementFailure:
    def test_impossible_constraints_raise(self):
        # 500 объектов площадью до 5 км² каждый в районе 10x10 км — неразрешимо
        cfg = make_config(
            district__width_km=10.0,
            district__height_km=10.0,
            objects__count=500,
            objects__area_max_m2=5_000_000.0,
            placement__max_attempts_per_object=50,
        )
        with pytest.raises(PlacementError) as exc_info:
            generate_scenario(cfg)
        assert "OBJ-" in str(exc_info.value)


class TestFrames:
    def test_frames_built(self, scenario):
        cfg = scenario.config.frames
        assert len(scenario.frames) == len(scenario.objects) * len(cfg.pass_azimuths_deg)
        for f in scenario.frames:
            assert 0.0 <= f.coverage_fraction <= 1.0
            assert f.object_id in {o.object_id for o in scenario.objects}
            ex = f.geometry.exterior.coords
            # стороны прямоугольника соответствуют размерам кадра
            from math import hypot
            d = [hypot(ex[i + 1][0] - ex[i][0], ex[i + 1][1] - ex[i][1]) for i in range(4)]
            along, cross = cfg.along_track_km * 1000, cfg.cross_track_km * 1000
            assert any(abs(v - along) < 1e-6 for v in d) or any(abs(v - cross) < 1e-6 for v in d)

    def test_frames_disabled(self):
        cfg = make_config(frames__enabled=False)
        sc = generate_scenario(cfg)
        assert sc.frames == []

    def test_large_object_partial_coverage(self):
        # объект больше кадра — coverage_fraction < 1
        from sro_generator.config import ManualCenter
        cfg = make_config(
            objects__count=1,
            objects__vertices_min=3,
            objects__vertices_max=3,
            objects__area_min_m2=20_000_000.0,
            objects__area_max_m2=20_000_000.0,
            placement__min_distance_m=0.0,
            placement__boundary_margin_m=0.0,
            frames__along_track_km=5.0,
            frames__cross_track_km=5.0,
        )
        cfg.placement.manual_centers = [ManualCenter(object_id="OBJ-001", latitude=55.0, longitude=60.0)]
        sc = generate_scenario(cfg)
        assert all(f.coverage_fraction < 1.0 for f in sc.frames)


class TestExport:
    def test_export_files_and_roundtrip(self, scenario, tmp_path):
        paths = export_scenario(scenario, tmp_path)
        for name in ("district.geojson", "objects.geojson", "objects.csv",
                     "frames.geojson", "config.resolved.json", "generation_report.json", "preview.png"):
            assert name in paths, f"нет файла {name}"
            assert paths[name].exists()

        objs = json.loads(paths["objects.geojson"].read_text(encoding="utf-8"))
        assert objs["type"] == "FeatureCollection"
        assert len(objs["features"]) == 25
        for feat in objs["features"]:
            geom = shape(feat["geometry"])
            assert geom.is_valid
            assert feat["properties"]["object_id"].startswith("OBJ-")
            lon, lat = geom.exterior.coords[0]
            assert -180 <= lon <= 180 and -90 <= lat <= 90  # порядок GeoJSON: lon, lat

        frames = json.loads(paths["frames.geojson"].read_text(encoding="utf-8"))
        assert len(frames["features"]) == len(scenario.frames)

        report = json.loads(paths["generation_report.json"].read_text(encoding="utf-8"))
        assert report["checks"]["count_matches_config"]
        assert report["checks"]["all_polygons_valid"]
        assert report["checks"]["areas_within_range"]

        csv_text = paths["objects.csv"].read_text(encoding="utf-8")
        assert "object_id" in csv_text.splitlines()[0]

        assert paths["preview.png"].stat().st_size > 0
