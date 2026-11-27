"""Тесты статистического исследования объектов съёмки (вкладка 4):
выборки расстояний до ближайшего соседа, имитационный эталон, шесть графиков."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from sro_generator.target_model import (
    MODELS,
    TARGET_MODELS,
    SpatialModelParams,
    collect_nn_samples,
    curve_coordinate_stats,
    generate_target_objects,
    nn_distance_sample,
    rayleigh_cdf,
    render_statistics_charts,
    run_curved_line_study,
    run_target_model_study,
    sample_skewness,
    uniform_ks_stat,
    write_curved_line_csv,
    write_curved_line_json,
)

DISTRICT = box(-12_500.0, -12_500.0, 12_500.0, 12_500.0)


def small_params(model: str = "uniform", count: int = 25) -> SpatialModelParams:
    return SpatialModelParams(
        model=model, count=count,
        area_min_m2=50_000.0, area_max_m2=300_000.0,
        boundary_margin_m=200.0, n_clusters=4, cluster_sigma_m=1_500.0,
        jitter_fraction=0.6, n_lines=3, line_sigma_m=300.0,
    )


class TestNnDistanceSample:
    @pytest.mark.parametrize("model", MODELS)
    def test_sample_nonempty_and_positive(self, model):
        r = generate_target_objects(small_params(model), DISTRICT, np.random.default_rng(0))
        d = nn_distance_sample(r)
        assert d.size >= 2
        assert np.all(d > 0.0)

    def test_too_few_objects(self):
        r = generate_target_objects(small_params(count=2), DISTRICT, np.random.default_rng(0))
        d = nn_distance_sample(r)
        assert d.size >= 1  # 2 объекта — одна пара расстояний, без краевой поправки

    def test_edge_correction_removes_boundary_bias(self):
        # для равномерного закона среднее по пулу с поправкой ближе к
        # теоретическому 0,5/√λ, чем без поправки
        lam = 60 / DISTRICT.area
        d_exp = 0.5 / math.sqrt(lam)
        pooled = []
        for i in range(20):
            r = generate_target_objects(small_params(count=60), DISTRICT, np.random.default_rng(i))
            pooled.append(nn_distance_sample(r))
        assert abs(np.concatenate(pooled).mean() - d_exp) / d_exp < 0.10


class TestCollectNnSamples:
    def test_pools_all_models(self):
        out = collect_nn_samples(small_params(), DISTRICT, models=TARGET_MODELS, realizations=5)
        assert set(out) == set(TARGET_MODELS)
        for v in out.values():
            assert isinstance(v, np.ndarray) and v.size >= 5 * 2

    def test_reproducible(self):
        a = collect_nn_samples(small_params(), DISTRICT, models=("uniform",), realizations=5)
        b = collect_nn_samples(small_params(), DISTRICT, models=("uniform",), realizations=5)
        np.testing.assert_array_equal(a["uniform"], b["uniform"])

    def test_reference_differs(self):
        emp = collect_nn_samples(small_params(), DISTRICT, models=("uniform",),
                                 realizations=5, seed_start=0)
        ref = collect_nn_samples(small_params(), DISTRICT, models=("uniform",),
                                 realizations=5, seed_start=100_000)
        assert not np.array_equal(emp["uniform"], ref["uniform"])

    def test_validation(self):
        with pytest.raises(ValueError):
            collect_nn_samples(small_params(), DISTRICT, realizations=0)


class TestCurvedLineStatistics:
    def test_uniform_ks_stat(self):
        # Равномерная сетка на [0,1] даёт известный D = 1/(2N).
        stat = uniform_ks_stat(np.linspace(0.05, 0.95, 10))
        assert stat["ks"] == pytest.approx(0.05)
        assert not stat["ks_rejected"]

    def test_curve_coordinate_stats_keys_and_theory(self):
        pooled = []
        for seed in range(30):
            p = SpatialModelParams(
                model="curved", count=20,
                area_min_m2=50_000.0, area_max_m2=300_000.0,
                boundary_margin_m=200.0, min_distance_m=80.0,
                curve_bend_m=1_500.0, curve_width_m=800.0,
            )
            r = generate_target_objects(p, DISTRICT, np.random.default_rng(seed))
            m = curve_coordinate_stats(r)
            assert 0.0 <= m["along_mean"] <= 1.0
            assert 0.0 <= m["along_ks"] <= 1.0
            assert 0.0 <= m["lateral_abs_mean_m"] <= 400.0
            pooled.append(m)
        abs_mean = float(np.mean([m["lateral_abs_mean_m"] for m in pooled]))
        # Теория для U[−w/2,w/2]: E|x| = w/4. Ограничения допускают умеренное смещение.
        assert abs_mean == pytest.approx(200.0, rel=0.20)
        along_ks = float(np.mean([m["along_ks"] for m in pooled]))
        assert along_ks < 0.25

    def test_run_curved_line_study(self):
        p = SpatialModelParams(
            model="curved", count=15,
            area_min_m2=50_000.0, area_max_m2=300_000.0,
            boundary_margin_m=200.0, min_distance_m=80.0,
            curve_width_m=900.0,
        )
        study = run_curved_line_study(
            p, DISTRICT, bend_values=[0.0, 1_500.0], realizations=12,
            seed_start=5, beta=0.95,
        )
        assert study["mode"] == "curved_line_statistics"
        assert study["beta"] == 0.95
        assert len(study["per_bend"]) == 2
        for item in study["per_bend"]:
            assert len(item["runs"]) == 12
            assert len(item["classical"]["nn_mean_m"]) == 11       # n = 2…12
            assert item["uniformity"]["lateral_std_theory_m"] == pytest.approx(900.0 / math.sqrt(12.0))
            last = item["classical"]["along_ks"][-1]
            assert last["ci_low"] < last["m_star"] < last["ci_high"]

    def test_curved_line_study_reproducible(self):
        p = SpatialModelParams(model="curved", count=10, curve_width_m=600.0)
        a = run_curved_line_study(p, DISTRICT, bend_values=[1_000.0], realizations=8,
                                  seed_start=0, beta=0.95)
        b = run_curved_line_study(p, DISTRICT, bend_values=[1_000.0], realizations=8,
                                  seed_start=0, beta=0.95)
        assert a["per_bend"][0]["classical"]["nn_mean_m"][-1]["m_star"] == \
               b["per_bend"][0]["classical"]["nn_mean_m"][-1]["m_star"]

    def test_main_study_contains_curve_diagnostics(self):
        study = run_target_model_study(
            small_params(count=8), DISTRICT, realizations=8, seed_start=0, beta=0.95,
            models=("curved",),
        )
        assert study["models"] == ["curved"]
        assert "along_ks" in study["curve_classical"]
        run = study["per_model"]["curved"]["runs"][0]
        for key in ("curve_length_m", "along_chi2", "lateral_std_m", "lateral_abs_mean_m"):
            assert key in run

    def test_write_curved_line_outputs(self, tmp_path):
        p = SpatialModelParams(model="curved", count=8, curve_width_m=500.0)
        study = run_curved_line_study(p, DISTRICT, bend_values=[0.0, 1_000.0],
                                      realizations=5, seed_start=0, beta=0.95)
        paths = write_curved_line_csv(study, tmp_path)
        assert set(paths) == {
            "curved_line_runs.csv", "curved_line_summary.csv", "curved_line_uniformity.csv",
        }
        for path in paths.values():
            assert path.exists() and path.stat().st_size > 50
        import json

        jpath = write_curved_line_json(study, tmp_path)
        data = json.loads(jpath.read_text(encoding="utf-8"))
        assert data["mode"] == "curved_line_statistics"
        assert len(data["per_bend"]) == 2


class TestSkewness:
    def test_symmetric_near_zero(self):
        rng = np.random.default_rng(0)
        assert abs(sample_skewness(rng.normal(0, 1, 5000))) < 0.1

    def test_right_skewed_positive(self):
        rng = np.random.default_rng(0)
        assert sample_skewness(rng.exponential(1.0, 5000)) > 1.0

    def test_small_sample_nan(self):
        assert math.isnan(sample_skewness([1.0, 2.0]))


class TestRayleighCdf:
    def test_bounds_and_monotonic(self):
        lam = 60 / (625e6)
        r = np.linspace(0, 10_000, 100)
        f = rayleigh_cdf(r, lam)
        assert f[0] == pytest.approx(0.0)
        assert np.all(np.diff(f) > 0)
        assert f[-1] < 1.0

    def test_median_matches_theory(self):
        lam = 1e-6
        r_med = math.sqrt(math.log(2) / (math.pi * lam))
        assert rayleigh_cdf(np.array([r_med]), lam)[0] == pytest.approx(0.5)


class TestStatisticsCharts:
    @pytest.fixture(scope="class")
    def artifacts(self):
        params = small_params(count=15)
        study = run_target_model_study(params, DISTRICT, realizations=10,
                                       seed_start=0, beta=0.95)
        nn_emp = collect_nn_samples(params, DISTRICT, realizations=10, seed_start=0)
        nn_ref = collect_nn_samples(params, DISTRICT, realizations=10, seed_start=100_000)
        examples = {m: generate_target_objects(
            SpatialModelParams(**study["per_model"][m]["params"]), DISTRICT,
            np.random.default_rng(0)) for m in TARGET_MODELS}
        return study, examples, nn_emp, nn_ref

    def test_six_charts_png_svg(self, artifacts, tmp_path):
        study, examples, nn_emp, nn_ref = artifacts
        paths = render_statistics_charts(
            study, examples, nn_emp, nn_ref,
            selected_model="uniform", metric="nn_mean_m",
            beta=0.95, epsilon0=50.0, out_dir=tmp_path, formats=("png", "svg"))
        assert len(paths) == 6
        for name, ps in paths.items():
            assert len(ps) == 2
            for p in ps:
                assert p.exists() and p.stat().st_size > 5_000

    def test_png_only(self, artifacts, tmp_path):
        study, examples, nn_emp, nn_ref = artifacts
        paths = render_statistics_charts(
            study, examples, nn_emp, nn_ref,
            selected_model="cluster", metric="ce_r",
            beta=0.95, epsilon0=0.05, out_dir=tmp_path, formats=("png",))
        assert len(paths) == 6
        assert all(len(ps) == 1 for ps in paths.values())

    def test_bad_model_or_metric(self, artifacts, tmp_path):
        study, examples, nn_emp, nn_ref = artifacts
        with pytest.raises(ValueError, match="закон"):
            render_statistics_charts(study, examples, nn_emp, nn_ref,
                                     selected_model="spiral", metric="ce_r",
                                     beta=0.95, epsilon0=0.05, out_dir=tmp_path)
        with pytest.raises(ValueError, match="характеристик"):
            render_statistics_charts(study, examples, nn_emp, nn_ref,
                                     selected_model="uniform", metric="nope",
                                     beta=0.95, epsilon0=0.05, out_dir=tmp_path)
