"""Тесты эквивалентности STRtree и полного перебора, повёрнутых районов и воспроизводимости."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sro_generator.config import ScenarioConfig
from sro_generator.generator import generate_scenario
from sro_generator.placement import PlacementError, place_objects
from sro_generator.geometry import LocalCRS, build_district_polygon

ROOT = Path(__file__).resolve().parent.parent


def make_config(**overrides) -> ScenarioConfig:
    data = json.loads((ROOT / "example_config.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        parts = key.split("__")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return ScenarioConfig.model_validate(data)


def fingerprint(scenario) -> tuple:
    """Детерминированный след результата для сравнения режимов размещения."""
    return (
        len(scenario.objects),
        tuple((o.object_id, round(o.area_m2, 6), round(o.center.x, 6), round(o.center.y, 6),
               o.vertex_count, round(o.rotation_deg, 6), o.priority, o.deadline_h, o.resolution_m)
              for o in scenario.objects),
        scenario.report.attempts_total,
    )


class TestStrtreeEquivalence:
    """Решения (принять/отклонить) STRtree и полного перебора совпадают."""

    @pytest.mark.parametrize("seed", [1, 7, 42, 1234])
    @pytest.mark.parametrize("min_distance,allow_overlap,max_frac", [
        (0.0, False, 0.0),
        (500.0, False, 0.0),
        (0.0, True, 0.4),
        (500.0, True, 0.6),
    ])
    def test_tree_matches_brute_force(self, seed, min_distance, allow_overlap, max_frac):
        cfg = make_config(
            seed=seed,
            objects__count=30,
            placement__min_distance_m=min_distance,
            placement__allow_overlap=allow_overlap,
            placement__max_overlap_fraction=max_frac if allow_overlap else 0.0,
        )
        with_tree = generate_scenario(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["placement"]["spatial_index"] = False
        brute = generate_scenario(ScenarioConfig.model_validate(cfg_dict))
        assert fingerprint(with_tree) == fingerprint(brute)
        # STRtree не пропускает конфликты: ограничения выполнены в обоих режимах
        for i, a in enumerate(with_tree.objects):
            for b in with_tree.objects[i + 1:]:
                inter = a.geometry.intersection(b.geometry).area
                if inter > 1e-12 * min(a.geometry.area, b.geometry.area):
                    assert allow_overlap
                    assert inter / min(a.geometry.area, b.geometry.area) <= max_frac + 1e-12
                else:
                    assert a.geometry.distance(b.geometry) >= min_distance - 1e-9

    def test_tree_reduces_exact_checks(self):
        """На плотных сценариях STRtree выполняет меньше точных проверок пар."""
        cfg = make_config(objects__count=120, district__width_km=40.0, district__height_km=40.0)
        with_tree = generate_scenario(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["placement"]["spatial_index"] = False
        brute = generate_scenario(ScenarioConfig.model_validate(cfg_dict))
        assert fingerprint(with_tree) == fingerprint(brute)
        assert with_tree.report.tree_queries > 0
        assert with_tree.report.tree_rebuilds > 0
        assert with_tree.report.exact_checks_total < brute.report.exact_checks_total

    def test_tree_used_for_manual_centers_too(self):
        from sro_generator.config import ManualCenter

        cfg = make_config(objects__count=25, district__width_km=30.0, district__height_km=30.0)
        cfg.placement.manual_centers = [ManualCenter(object_id="OBJ-001", latitude=55.0, longitude=60.0)]
        scenario = generate_scenario(cfg)
        assert scenario.report.tree_queries > 0


class TestRotatedDistricts:
    @pytest.mark.parametrize("rotation", [0.0, 15.0, 37.0, 90.0, 135.0, 210.0])
    def test_objects_stay_inside_with_margin(self, rotation):
        cfg = make_config(district__rotation_deg=rotation, objects__count=20)
        scenario = generate_scenario(cfg)
        inner = scenario.district_local.buffer(-cfg.placement.boundary_margin_m)
        for o in scenario.objects:
            assert inner.covers(o.geometry)
        # центры внутри допустимых областей: вершины не выходят за границы района
        for o in scenario.objects:
            assert scenario.district_local.covers(o.geometry)

    def test_rotated_legacy_mode_works(self):
        cfg = make_config(district__rotation_deg=45.0, objects__count=10,
                          placement__center_region_mode="legacy")
        scenario = generate_scenario(cfg)
        assert len(scenario.objects) == 10


class TestReproducibility11:
    def test_same_seed_same_result(self):
        s1 = generate_scenario(make_config(seed=777))
        s2 = generate_scenario(make_config(seed=777))
        assert fingerprint(s1) == fingerprint(s2)

    def test_seed_changes_result(self):
        s1 = generate_scenario(make_config(seed=777))
        s2 = generate_scenario(make_config(seed=778))
        assert fingerprint(s1) != fingerprint(s2)

    def test_exact_mode_reproducible_across_region_modes_is_not_required(self):
        """Режимы области центров могут давать разные (валидные) результаты —
        документированное изменение 1.1.0; внутри режима воспроизводимость полная."""
        exact1 = generate_scenario(make_config(seed=5))
        exact2 = generate_scenario(make_config(seed=5))
        assert fingerprint(exact1) == fingerprint(exact2)


class TestMarginLimits:
    def test_margin_equal_half_width_fails(self):
        cfg = make_config(district__width_km=10.0, district__height_km=10.0,
                          placement__boundary_margin_m=5000.0, objects__count=1)
        with pytest.raises(PlacementError):
            generate_scenario(cfg)

    def test_margin_just_below_limit_works(self):
        cfg = make_config(district__width_km=10.0, district__height_km=10.0,
                          placement__boundary_margin_m=4900.0,
                          objects__area_max_m2=1000.0, objects__area_min_m2=500.0,
                          objects__count=1, placement__min_distance_m=0.0)
        scenario = generate_scenario(cfg)
        assert len(scenario.objects) == 1


class TestRegionModeConfig:
    def test_defaults(self):
        cfg = make_config()
        assert cfg.placement.center_region_mode == "exact"
        assert cfg.placement.spatial_index is True

    def test_legacy_mode_keyword_rejected(self):
        with pytest.raises(Exception):
            make_config(placement__center_region_mode="other")

    def test_place_objects_kwargs_override_config(self):
        cfg = make_config(objects__count=5)
        local = LocalCRS(cfg.district.center_lat, cfg.district.center_lon)
        district = build_district_polygon(local, cfg.district.width_km * 1000.0,
                                          cfg.district.height_km * 1000.0, cfg.district.rotation_deg)
        rng = __import__("numpy").random.default_rng(3)
        placed, report = place_objects(
            rng, district, cfg.placement, cfg.objects, cfg.attributes, [], local.to_xy,
            use_spatial_index=False, center_region_mode="legacy",
        )
        assert len(placed) == 5
        assert report.tree_queries == 0
