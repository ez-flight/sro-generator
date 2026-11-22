"""Тесты статистической корректности (версия 1.1.0): χ², KS, NND, воспроизводимость."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sro_generator.stats import (
    QUADRANT_GRID,
    chi2_gof,
    chi2_sf,
    clark_evans_index,
    kolmogorov_sf,
    ks_test,
    run_batch,
    simulate_reference_quadrant_counts,
)

ROOT = Path(__file__).resolve().parent.parent


def make_config(**overrides):
    data = json.loads((ROOT / "example_config.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        parts = key.split("__")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return data


class TestChi2Gof:
    def test_perfect_fit(self):
        r = chi2_gof([10, 10, 10, 10], [10.0] * 4)
        assert r["statistic"] == 0.0
        assert r["p_value"] == 1.0
        assert r["df"] == 3
        assert r["applicable"]
        assert r["warnings"] == []

    def test_known_statistic(self):
        r = chi2_gof([5, 15], [10.0, 10.0])
        assert r["statistic"] == pytest.approx(5.0)
        assert r["p_value"] == pytest.approx(chi2_sf(5.0, 1))

    def test_strong_deviation_small_p(self):
        r = chi2_gof([0, 40], [20.0, 20.0])
        assert r["statistic"] == 40.0
        assert r["p_value"] < 1e-9

    def test_low_expected_warning(self):
        r = chi2_gof([1, 2, 3], [2.0, 2.0, 2.0])
        assert any("ожидаем" in w or "частот" in w for w in r["warnings"])
        assert r["min_expected"] == 2.0

    def test_zero_expected_bins_dropped(self):
        r = chi2_gof([10, 10, 0], [10.0, 10.0, 0.0])
        assert r["df"] == 1
        assert any("исключены" in w for w in r["warnings"])

    def test_total_mismatch_rescaled_with_warning(self):
        r = chi2_gof([15, 5], [5.0, 5.0])
        assert any("перенормированы" in w for w in r["warnings"])
        assert r["statistic"] == pytest.approx(5.0)  # ожидание перенормировано к [10, 10]

    def test_validation(self):
        with pytest.raises(ValueError):
            chi2_gof([1, 2], [1.0])
        with pytest.raises(ValueError):
            chi2_gof([1, 2], [1.0, -1.0])


class TestSpecialFunctions:
    def test_chi2_sf_known_quantiles(self):
        assert chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-5)
        assert chi2_sf(5.991465, 2) == pytest.approx(0.05, abs=1e-5)
        assert chi2_sf(9.487729, 4) == pytest.approx(0.05, abs=1e-5)
        assert chi2_sf(0.0, 5) == 1.0
        assert math.isnan(chi2_sf(1.0, 0))

    def test_kolmogorov_sf_limits(self):
        assert kolmogorov_sf(0.0) == 1.0
        assert kolmogorov_sf(2.0) == pytest.approx(0.0006709253, rel=1e-6)
        assert kolmogorov_sf(0.9) == pytest.approx(0.3927307079, rel=1e-6)


class TestKsTest:
    def test_uniform_sample_high_p(self):
        rng = np.random.default_rng(42)
        sample = rng.uniform(0, 1, 2000).tolist()
        r = ks_test(sample, lambda x: min(max(x, 0.0), 1.0))
        assert r["applicable"]
        assert r["p_value"] > 0.05
        assert r["statistic"] < 0.05

    def test_wrong_distribution_low_p(self):
        rng = np.random.default_rng(7)
        sample = rng.normal(0.5, 0.1, 2000).tolist()
        r = ks_test(sample, lambda x: min(max(x, 0.0), 1.0))
        assert r["p_value"] < 1e-6

    def test_small_sample_warning(self):
        r = ks_test([0.1, 0.2, 0.3, 0.4, 0.5], lambda x: x)
        assert any("n =" in w for w in r["warnings"])

    def test_ties_warning(self):
        r = ks_test([0.5, 0.5, 0.5, 0.2, 0.9, 0.1], lambda x: x)
        assert any("совпадающие" in w for w in r["warnings"])

    def test_discrete_not_applicable(self):
        r = ks_test([1, 2, 3, 4], lambda x: x / 4, discrete=True)
        assert not r["applicable"]
        assert any("дискретн" in w for w in r["warnings"])

    def test_degenerate_range_cdf(self):
        """Совпадающие пределы диапазона: cdf вырождается — предупреждение в summarize."""
        sample = [2.0, 2.0, 2.0, 2.0]
        r = ks_test(sample, lambda x: 0.0 if x < 2.0 else 1.0)
        assert r["kind"] == "ks"


@pytest.fixture(scope="module")
def batch():
    return run_batch(make_config(objects__count=15), n_runs=5, workers=2, seed_start=300)


class TestBatchStatistics:

    def test_statistical_tests_present(self, batch):
        tests = batch["summary"]["statistical_tests"]
        for key in ("vertices", "classes", "priority", "deadline", "resolution", "areas", "rotations", "quadrant_grid"):
            assert key in tests
            t = tests[key]
            assert "interpretation" in t
        for key in ("vertices", "classes", "priority", "deadline", "resolution"):
            assert "не доказательство равномерности" in tests[key]["interpretation"]
        assert "пространственного распределения" in tests["quadrant_grid"]["interpretation"]

    def test_chi2_tests_applicable_and_finite(self, batch):
        for key in ("vertices", "classes", "priority", "deadline", "resolution"):
            t = batch["summary"]["statistical_tests"][key]
            if t["applicable"]:
                assert 0.0 <= t["p_value"] <= 1.0
                assert t["statistic"] >= 0.0
                assert t["df"] >= 1

    def test_ks_tests_applicable(self, batch):
        for key in ("areas", "rotations"):
            t = batch["summary"]["statistical_tests"][key]
            assert t["kind"] == "ks"
            assert t["applicable"]
            assert 0.0 <= t["p_value"] <= 1.0

    def test_quadrant_uses_area_weighting(self, batch):
        assert batch["summary"]["quadrant_area_weighted"] is True
        t = batch["summary"]["statistical_tests"]["quadrant_grid"]
        assert any("площад" in w for w in t["warnings"])

    def test_clark_evans_descriptive(self, batch):
        ce = batch["summary"]["clark_evans_r"]
        assert ce["mean"] > 0.5
        assert any("min_distance" in w for w in batch["summary"]["clark_evans_warnings"])


class TestReproducibilityStats:
    def test_workers_do_not_change_results(self):
        seq = run_batch(make_config(objects__count=8), n_runs=4, workers=1, seed_start=55)
        par = run_batch(make_config(objects__count=8), n_runs=4, workers=3, seed_start=55)
        for r1, r2 in zip(seq["runs"], par["runs"]):
            assert r1["objects"] == r2["objects"]
            assert r1["attempts_total"] == r2["attempts_total"]
            assert r1["quadrant_counts"] == r2["quadrant_counts"]
            assert r1["grid_cell_area_fractions"] == r2["grid_cell_area_fractions"]
        assert seq["summary"]["statistical_tests"] == par["summary"]["statistical_tests"]

    def test_same_seed_same_batch(self):
        b1 = run_batch(make_config(objects__count=6), n_runs=3, workers=1, seed_start=88)
        b2 = run_batch(make_config(objects__count=6), n_runs=3, workers=1, seed_start=88)
        # duration_s — измерение времени, исключаем из сравнения
        for r1, r2 in zip(b1["runs"], b2["runs"]):
            d1 = {k: v for k, v in r1.items() if k != "duration_s"}
            d2 = {k: v for k, v in r2.items() if k != "duration_s"}
            assert d1 == d2

    def test_reference_quadrant_simulation(self):
        ref = simulate_reference_quadrant_counts(make_config(objects__count=10), n_sim=4, workers=1, seed_start=900)
        assert len(ref) == QUADRANT_GRID * QUADRANT_GRID
        assert all(c >= 0 for c in ref)
        batch = run_batch(make_config(objects__count=10), n_runs=4, workers=1, seed_start=910,
                          reference_quadrant_counts=ref)
        t = batch["summary"]["statistical_tests"]["quadrant_grid"]
        assert any("эталон" in w or "симуляц" in w for w in t["warnings"])

    def test_clark_evans_index_values(self):
        # для CSR mean(NND) ≈ 0.5·√(A/n): индекс около 1
        assert clark_evans_index(500.0, 1e8, 100) == pytest.approx(500.0 / 500.0, rel=1e-9)
        assert clark_evans_index(500.0, 1e8, 1) is None
        assert clark_evans_index(500.0, 0.0, 100) is None
