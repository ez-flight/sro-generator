"""Тесты совместной модели района ДЗЗ (объекты T + препятствия B)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from sro_generator.geometry import LocalCRS, build_district_polygon
from sro_generator.joint_model import (
    STATUS_HIDDEN,
    STATUS_PARTIAL,
    STATUS_VISIBLE,
    compute_joint_geometry,
    export_joint_geojson,
    generate_joint_model,
    joint_summary,
    render_joint_map,
)
from sro_generator.target_model import MODELS, SpatialModelParams, TargetObject, generate_target_objects

DISTRICT = box(-12_500.0, -12_500.0, 12_500.0, 12_500.0)


def target_params(model: str = "uniform", count: int = 30) -> SpatialModelParams:
    return SpatialModelParams(
        model=model, count=count,
        area_min_m2=50_000.0, area_max_m2=300_000.0,
        boundary_margin_m=200.0,
        n_clusters=4, cluster_sigma_m=1_500.0, jitter_fraction=0.6,
        n_lines=3, line_sigma_m=300.0,
    )


def obstacle_params(count: int = 5) -> SpatialModelParams:
    return SpatialModelParams(
        model="uniform", count=count,
        vertices_min=6, vertices_max=10,
        area_min_m2=20e6, area_max_m2=50e6,
        irregularity=0.5, boundary_margin_m=0.0, allow_overlap=True,
    )


def _square(cx: float, cy: float, half: float, oid: str) -> TargetObject:
    poly = box(cx - half, cy - half, cx + half, cy + half)
    return TargetObject(object_id=oid, geometry=poly, center=Point(cx, cy),
                        area_m2=poly.area, vertex_count=4, rotation_deg=0.0)


# ---------------------------------------------------------------------------
# Чистая геометрия: контрольные примеры с аналитическим ответом
# ---------------------------------------------------------------------------

class TestJointGeometryControls:
    def test_no_obstacles_full_visibility(self):
        targets = [_square(0, 0, 100, "T1"), _square(1000, 0, 100, "T2")]
        g = compute_joint_geometry(targets, [])
        assert g.p_opt == pytest.approx(1.0)
        assert g.p_skr == pytest.approx(0.0)
        assert g.area_reshoot_m2 == pytest.approx(0.0)
        assert all(r["status"] == STATUS_VISIBLE for r in g.per_object)

    def test_full_cover_all_hidden(self):
        targets = [_square(0, 0, 100, "T1"), _square(1000, 0, 100, "T2")]
        obstacles = [_square(500, 0, 5000, "B1")]  # покрывает весь район объектов
        g = compute_joint_geometry(targets, obstacles)
        assert g.p_skr == pytest.approx(1.0)
        assert g.p_opt == pytest.approx(0.0)
        assert all(r["status"] == STATUS_HIDDEN for r in g.per_object)
        assert g.area_reshoot_m2 == pytest.approx(g.area_targets_m2)

    def test_half_cover_analytic(self):
        # объект 200×200, препятствие накрывает ровно половину (x > 0)
        targets = [_square(0, 0, 100, "T1")]
        obstacles = [_square(150, 0, 150, "B1")]  # box(0,-150,300,150) ∩ объект = x∈[0,100]
        g = compute_joint_geometry(targets, obstacles)
        assert g.p_skr == pytest.approx(0.5)
        assert g.p_opt == pytest.approx(0.5)
        assert g.per_object[0]["status"] == STATUS_PARTIAL
        assert g.per_object[0]["hidden_area_m2"] == pytest.approx(20_000.0)

    def test_area_identity_holds(self):
        targets = [_square(i * 300.0, (i % 3) * 300.0, 100, f"T{i}") for i in range(9)]
        obstacles = [_square(400, 400, 500, "B1"), _square(-200, 100, 300, "B2")]
        g = compute_joint_geometry(targets, obstacles)
        assert g.area_reshoot_m2 + g.area_visible_m2 == pytest.approx(
            g.area_targets_m2, rel=1e-9)
        assert g.p_opt + g.p_skr == pytest.approx(1.0)

    def test_disjoint_obstacle_no_effect(self):
        targets = [_square(0, 0, 100, "T1")]
        obstacles = [_square(10_000, 10_000, 500, "B1")]
        g = compute_joint_geometry(targets, obstacles)
        assert g.p_skr == pytest.approx(0.0)
        assert g.per_object[0]["status"] == STATUS_VISIBLE

    def test_requires_targets(self):
        with pytest.raises(ValueError, match="объект"):
            compute_joint_geometry([], [])


# ---------------------------------------------------------------------------
# Генерация совместной модели
# ---------------------------------------------------------------------------

class TestJointGeneration:
    @pytest.mark.parametrize("model", MODELS)
    def test_all_target_laws(self, model):
        r = generate_joint_model(target_params(model), obstacle_params(),
                                 DISTRICT, target_seed=1, obstacle_seed=2)
        assert len(r.targets.objects) == 30
        assert len(r.obstacles.objects) == 5
        g = r.geometry
        assert 0.0 <= g.p_skr <= 1.0
        assert g.p_opt + g.p_skr == pytest.approx(1.0)
        assert g.area_reshoot_m2 + g.area_visible_m2 == pytest.approx(
            g.area_targets_m2, rel=1e-9)
        # геометрия внутри района
        for o in r.obstacles.objects:
            assert DISTRICT.covers(o.geometry)

    def test_reproducibility(self):
        a = generate_joint_model(target_params(), obstacle_params(), DISTRICT, 5, 6)
        b = generate_joint_model(target_params(), obstacle_params(), DISTRICT, 5, 6)
        assert [o.center.x for o in a.targets.objects] == \
               [o.center.x for o in b.targets.objects]
        assert [o.center.x for o in a.obstacles.objects] == \
               [o.center.x for o in b.obstacles.objects]
        assert a.geometry.p_skr == b.geometry.p_skr

    def test_independent_seeds(self):
        base = generate_joint_model(target_params(), obstacle_params(), DISTRICT, 5, 6)
        # меняем seed препятствий — объекты не меняются
        r_b = generate_joint_model(target_params(), obstacle_params(), DISTRICT, 5, 7)
        assert [o.center.x for o in base.targets.objects] == \
               [o.center.x for o in r_b.targets.objects]
        assert [o.center.x for o in base.obstacles.objects] != \
               [o.center.x for o in r_b.obstacles.objects]
        # меняем seed объектов — препятствия не меняются
        r_t = generate_joint_model(target_params(), obstacle_params(), DISTRICT, 6, 6)
        assert [o.center.x for o in base.obstacles.objects] == \
               [o.center.x for o in r_t.obstacles.objects]
        assert [o.center.x for o in base.targets.objects] != \
               [o.center.x for o in r_t.targets.objects]

    def test_no_obstacles_mode(self):
        r = generate_joint_model(target_params(), None, DISTRICT, 1, 2)
        assert r.obstacles is None
        assert r.geometry.p_skr == 0.0
        assert r.geometry.p_opt == 1.0

    def test_classification_consistency(self):
        r = generate_joint_model(target_params(count=40), obstacle_params(count=6),
                                 DISTRICT, 3, 4)
        rows = r.geometry.per_object
        for row in rows:
            assert row["hidden_area_m2"] + row["visible_area_m2"] == pytest.approx(
                row["area_m2"], rel=1e-9)
            if row["status"] == STATUS_VISIBLE:
                assert row["hidden_fraction"] <= 1e-9
            elif row["status"] == STATUS_HIDDEN:
                assert row["hidden_fraction"] == 1.0
            else:
                assert 1e-9 < row["hidden_fraction"] < 1.0 - 1e-6
        s = joint_summary(r)
        assert sum(s["counts"].values()) == len(rows)
        assert s["area_identity_ok"]

    def test_overlap_option_for_obstacles(self):
        # с allow_overlap=True перекрытия разрешены; без него большие поля не разместятся
        p = obstacle_params(count=8)
        r = generate_joint_model(target_params(count=10), p, DISTRICT, 1, 2)
        unions = sum(o.area_m2 for o in r.obstacles.objects)
        union_area = r.geometry.obstacle_union.area
        assert union_area <= unions + 1e-6  # объединение не больше суммы (перекрытия слиты)


# ---------------------------------------------------------------------------
# Экспорт и карта
# ---------------------------------------------------------------------------

class TestJointExport:
    @pytest.fixture()
    def joint(self):
        local = LocalCRS(55.0, 60.0)
        district = build_district_polygon(local, 25_000.0, 25_000.0, 0.0)
        r = generate_joint_model(target_params(count=20), obstacle_params(count=4),
                                 district, 1, 2)
        return r, local

    def test_geojson_layers(self, joint):
        r, local = joint
        layers = export_joint_geojson(r, local)
        assert set(layers) == {"district", "objects", "obstacles", "reshoot", "visible"}
        assert len(layers["district"]["features"]) == 1
        assert len(layers["objects"]["features"]) == 20
        assert len(layers["obstacles"]["features"]) == 4
        assert len(layers["reshoot"]["features"]) >= 1
        f0 = layers["objects"]["features"][0]
        assert f0["geometry"]["type"] == "Polygon"
        assert "hidden_fraction" in f0["properties"]
        assert f0["properties"]["status"] in (STATUS_VISIBLE, STATUS_PARTIAL, STATUS_HIDDEN)
        # WGS 84
        lon, lat = f0["geometry"]["coordinates"][0][0]
        assert 59.0 < lon < 61.0 and 54.0 < lat < 56.0
        # площади частей reshoot согласованы с метрикой
        s_parts = sum(f["properties"]["area_m2"] for f in layers["reshoot"]["features"])
        assert s_parts == pytest.approx(r.geometry.area_reshoot_m2, rel=1e-6)

    def test_summary_json_serializable(self, joint):
        import json

        r, _ = joint
        s = joint_summary(r)
        json.dumps(s)  # не должно падать
        assert s["mode"] == "joint_spatial_model"

    def test_render_map_layers(self, joint, tmp_path):
        r, _ = joint
        p_all = render_joint_map(r, tmp_path / "all.png")
        p_min = render_joint_map(r, tmp_path / "min.png",
                                 layers={"obstacles": False, "visible": False,
                                         "reshoot": False})
        assert p_all.exists() and p_min.exists()
        assert p_all.stat().st_size > 20_000
