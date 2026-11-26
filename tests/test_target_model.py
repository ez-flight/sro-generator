"""Тесты модуля модели объектов съёмки (законы пространственного размещения)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from sro_generator.placement import PlacementError
from sro_generator.target_model import (
    MODELS,
    SpatialModelParams,
    chi2_quantile,
    generate_target_objects,
    nearest_neighbor_stats,
    quadrant_stats,
    realization_metrics,
    run_target_model_study,
    write_target_csv,
    write_target_json,
    render_target_charts,
    export_target_geojson,
)

DISTRICT = box(-12_500.0, -12_500.0, 12_500.0, 12_500.0)  # 25×25 км


def small_params(model: str, count: int = 25) -> SpatialModelParams:
    return SpatialModelParams(
        model=model, count=count,
        area_min_m2=50_000.0, area_max_m2=300_000.0,
        boundary_margin_m=200.0, min_distance_m=100.0,
        n_clusters=4, cluster_sigma_m=1_500.0, jitter_fraction=0.6,
        n_lines=3, line_sigma_m=300.0,
    )


# ---------------------------------------------------------------------------
# Валидация параметров
# ---------------------------------------------------------------------------

class TestParamsValidation:
    def test_bad_model(self):
        with pytest.raises(ValueError, match="model"):
            SpatialModelParams(model="spiral")

    def test_bad_count(self):
        with pytest.raises(ValueError, match="count"):
            SpatialModelParams(count=0)

    def test_bad_vertices(self):
        with pytest.raises(ValueError, match="vertices"):
            SpatialModelParams(vertices_min=2)

    def test_bad_areas(self):
        with pytest.raises(ValueError, match="area"):
            SpatialModelParams(area_min_m2=10.0, area_max_m2=5.0)

    def test_bad_jitter(self):
        with pytest.raises(ValueError, match="jitter"):
            SpatialModelParams(jitter_fraction=1.5)

    def test_bad_clusters(self):
        with pytest.raises(ValueError, match="n_clusters"):
            SpatialModelParams(n_clusters=0)

    def test_bad_lines(self):
        with pytest.raises(ValueError, match="n_lines"):
            SpatialModelParams(n_lines=0)


# ---------------------------------------------------------------------------
# Размещение: инварианты для всех законов
# ---------------------------------------------------------------------------

class TestPlacement:
    @pytest.mark.parametrize("model", MODELS)
    def test_count_and_bounds(self, model):
        p = small_params(model)
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        assert len(r.objects) == p.count
        inner = DISTRICT.buffer(-p.boundary_margin_m)
        for o in r.objects:
            assert inner.covers(o.geometry)
            assert o.geometry.is_valid and o.geometry.area > 0.0

    @pytest.mark.parametrize("model", MODELS)
    def test_no_overlap_and_min_distance(self, model):
        p = small_params(model)
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(1))
        for i, a in enumerate(r.objects):
            for b in r.objects[i + 1:]:
                assert a.geometry.intersection(b.geometry).area <= 1e-9
                assert a.geometry.distance(b.geometry) >= p.min_distance_m - 1e-9

    @pytest.mark.parametrize("model", MODELS)
    def test_reproducibility(self, model):
        p = small_params(model)
        r1 = generate_target_objects(p, DISTRICT, np.random.default_rng(42))
        r2 = generate_target_objects(p, DISTRICT, np.random.default_rng(42))
        assert len(r1.objects) == len(r2.objects)
        for a, b in zip(r1.objects, r2.objects):
            assert a.center.x == b.center.x and a.center.y == b.center.y
            assert a.area_m2 == b.area_m2

    @pytest.mark.parametrize("model", MODELS)
    def test_different_seeds_differ(self, model):
        p = small_params(model)
        r1 = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        r2 = generate_target_objects(p, DISTRICT, np.random.default_rng(1))
        c1 = [(o.center.x, o.center.y) for o in r1.objects]
        c2 = [(o.center.x, o.center.y) for o in r2.objects]
        assert c1 != c2

    def test_area_range(self):
        p = small_params("uniform")
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        for o in r.objects:
            assert 0.99 * p.area_min_m2 <= o.area_m2 <= 1.01 * p.area_max_m2

    def test_margin_too_big(self):
        p = small_params("uniform")
        p.boundary_margin_m = 20_000.0
        with pytest.raises(PlacementError, match="отступ"):
            generate_target_objects(p, DISTRICT, np.random.default_rng(0))

    def test_cluster_groups_assigned(self):
        p = small_params("cluster")
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        assert len(r.cluster_centers) == p.n_clusters
        assert all(o.group_id is not None and 0 <= o.group_id < p.n_clusters for o in r.objects)

    def test_linear_chords_present(self):
        p = small_params("linear")
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        assert 1 <= len(r.line_chords) <= p.n_lines
        assert all(o.group_id is not None and 0 <= o.group_id < len(r.line_chords)
                   for o in r.objects)

    def test_regular_nodes_inside(self):
        p = small_params("regular", count=20)
        r = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        assert len(r.objects) == 20


# ---------------------------------------------------------------------------
# Пространственные статистики
# ---------------------------------------------------------------------------

class TestSpatialStats:
    def test_nn_stats_grid(self):
        # идеальная решётка 5×5 с шагом 1000 м
        pts = np.array([(i * 1000.0, j * 1000.0) for i in range(5) for j in range(5)])
        s = nearest_neighbor_stats(pts, 25_000_000.0)
        assert s["nn_mean_m"] == pytest.approx(1000.0)
        assert s["ce_r"] > 1.0  # решётка регулярнее случайного

    def test_nn_stats_requires_two(self):
        with pytest.raises(ValueError):
            nearest_neighbor_stats(np.array([[0.0, 0.0]]), 1e6)

    def test_chi2_quantile_vs_table(self):
        # табличные квантили χ² (df = 24): 12.401 и 39.364
        assert chi2_quantile(0.025, 24) == pytest.approx(12.401, abs=0.05)
        assert chi2_quantile(0.975, 24) == pytest.approx(39.364, abs=0.05)

    def test_chi2_quantile_validation(self):
        with pytest.raises(ValueError):
            chi2_quantile(0.5, 0)
        with pytest.raises(ValueError):
            chi2_quantile(1.5, 10)

    def test_quadrant_uniform_not_rejected(self):
        rng = np.random.default_rng(7)
        rejects = 0
        for _ in range(100):
            pts = np.column_stack([rng.uniform(-12_500, 12_500, 60),
                                   rng.uniform(-12_500, 12_500, 60)])
            rejects += quadrant_stats(pts, DISTRICT, grid=5)["csr_rejected"]
        assert rejects / 100 < 0.15  # номинал 5 %, допуск на флуктуации

    def test_quadrant_clustered_rejected(self):
        rng = np.random.default_rng(7)
        pts = np.column_stack([rng.normal(0.0, 800.0, 60), rng.normal(0.0, 800.0, 60)])
        q = quadrant_stats(pts, DISTRICT, grid=5)
        assert q["csr_rejected"] and q["pattern"] == "clustered"
        assert q["dispersion_index"] > 1.0

    def test_realization_metrics_keys(self):
        r = generate_target_objects(small_params("uniform"), DISTRICT, np.random.default_rng(0))
        m = realization_metrics(r)
        for key in ("nn_mean_m", "ce_r", "dispersion_index", "nn_bearing_r",
                    "objects_area_fraction", "density_per_km2", "quad_csr_rejected"):
            assert key in m
        assert 0.0 < m["objects_area_fraction"] < 1.0
        assert m["density_per_km2"] == pytest.approx(25 / 625.0)


# ---------------------------------------------------------------------------
# Различение законов (статистические сигнатуры)
# ---------------------------------------------------------------------------

class TestModelSignatures:
    @pytest.fixture(scope="class")
    def study(self):
        return run_target_model_study(
            small_params("uniform", count=40), DISTRICT,
            realizations=40, seed_start=100, beta=0.95,
        )

    def test_uniform_close_to_csr(self, study):
        pm = study["per_model"]["uniform"]
        assert pm["ce_z_reject_share"] <= 0.25
        assert pm["csr_chi2_reject_share"] <= 0.25
        last = pm["classical"]["ce_r"][-1]
        assert last["ci_low"] <= 1.10 and last["ci_high"] >= 0.90

    def test_cluster_grouped(self, study):
        pm = study["per_model"]["cluster"]
        last_r = pm["classical"]["ce_r"][-1]
        last_i = pm["classical"]["dispersion_index"][-1]
        assert last_r["ci_high"] < 1.0        # группирование
        assert last_i["ci_low"] > 1.0
        assert pm["ce_z_reject_share"] > 0.8

    def test_regular_dispersed(self, study):
        pm = study["per_model"]["regular"]
        last_r = pm["classical"]["ce_r"][-1]
        last_i = pm["classical"]["dispersion_index"][-1]
        assert last_r["ci_low"] > 1.0         # регулярность
        assert last_i["ci_high"] < 1.0

    def test_linear_anisotropic(self, study):
        pm = study["per_model"]["linear"]
        uni = study["per_model"]["uniform"]
        lin_b = pm["classical"]["nn_bearing_r"][-1]["m_star"]
        uni_b = uni["classical"]["nn_bearing_r"][-1]["m_star"]
        assert lin_b > uni_b                  # выраженная анизотропия направлений

    def test_classical_series_consistency(self, study):
        for model in MODELS:
            series = study["per_model"][model]["classical"]["nn_mean_m"]
            assert len(series) == 40 - 1      # n = 2…N
            last = series[-1]
            assert last["ci_low"] < last["m_star"] < last["ci_high"]
            assert last["sigma_tilde"] == pytest.approx(math.sqrt(last["d_tilde"]))


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.fixture(scope="class")
    def study(self):
        return run_target_model_study(
            small_params("uniform", count=15), DISTRICT,
            realizations=10, seed_start=0, beta=0.95,
        )

    def test_csv(self, study, tmp_path):
        paths = write_target_csv(study, tmp_path)
        assert set(paths) == {"target_runs.csv", "target_summary.csv"}
        runs = (tmp_path / "target_runs.csv").read_text(encoding="utf-8").splitlines()
        assert len(runs) == 1 + 4 * 10

    def test_json(self, study, tmp_path):
        path = write_target_json(study, tmp_path)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mode"] == "target_object_distribution"
        assert set(data["per_model"]) == set(MODELS)

    def test_charts(self, study, tmp_path):
        examples = {}
        for model in MODELS:
            p = SpatialModelParams(**study["per_model"][model]["params"])
            examples[model] = generate_target_objects(p, DISTRICT, np.random.default_rng(0))
        paths = render_target_charts(study, examples, tmp_path)
        assert len(paths) == 4
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 10_000

    def test_geojson(self):
        from sro_generator.geometry import LocalCRS, build_district_polygon

        local = LocalCRS(55.0, 60.0)
        district = build_district_polygon(local, 25_000.0, 25_000.0, 0.0)
        r = generate_target_objects(small_params("cluster", count=10), district,
                                    np.random.default_rng(0))
        fc = export_target_geojson(r, local)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 10
        props = fc["features"][0]["properties"]
        assert props["model"] == "cluster"
        assert -90 <= props["center_lat"] <= 90
        assert fc["features"][0]["geometry"]["type"] == "Polygon"
