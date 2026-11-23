"""Тесты исследования сходимости итерационного алгоритма формирования кадров РСА."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.convergence import (
    ConvergenceParams,
    compute_iteration_stats,
    coverage_threshold_analysis,
    diagnose_convergence,
    render_comparison_charts,
    render_convergence_charts,
    run_convergence_study,
    run_experiment,
    std_trend_analysis,
    write_convergence_csv,
    write_convergence_json,
)

ROOT = Path(__file__).resolve().parent.parent


def make_config(**overrides) -> ScenarioConfig:
    data = json.loads((ROOT / "example_config.json").read_text(encoding="utf-8"))
    # уменьшенный сценарий для скорости: 8 объектов × 2 азимута = 16 кадров-кандидатов
    data["district"]["width_km"] = 30.0
    data["district"]["height_km"] = 30.0
    data["objects"]["count"] = 8
    data["objects"]["area_min_m2"] = 100_000.0
    data["objects"]["area_max_m2"] = 400_000.0
    for key, value in overrides.items():
        parts = key.split("__")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return ScenarioConfig.model_validate(data)


@pytest.fixture(scope="module")
def study():
    params = ConvergenceParams(experiments=4, iterations=8, seed_start=100)
    return run_convergence_study(make_config(), params, workers=1)


class TestStudyStructure:
    def test_shapes(self, study):
        assert len(study["experiments"]) == 4
        for e in study["experiments"]:
            assert len(e["trajectory"]) == 8
            assert len(e["district_coverage"]) == 8
            assert len(e["objects_coverage"]) == 8
            assert len(e["frames_selected"]) == 8
            assert len(e["marginal_gains"]) == 8
        assert len(study["iteration_stats"]) == 8

    def test_trajectories_monotone_and_bounded(self, study):
        for e in study["experiments"]:
            traj = e["trajectory"]
            for a, b in zip(traj, traj[1:]):
                assert b >= a - 1e-12  # объединение кадров не уменьшает покрытие
            for v in traj + e["district_coverage"] + e["objects_coverage"]:
                assert 0.0 <= v <= 1.0

    def test_increment_matches_trajectory(self, study):
        for e in study["experiments"]:
            for k in range(1, 8):
                expected = e["trajectory"][k] - e["trajectory"][k - 1]
                assert abs(e["marginal_gains"][k] - expected) < 1e-9

    def test_scenario_fixed_between_experiments(self, study):
        fp = study["experiments"][0]["scenario_fingerprint"]
        for e in study["experiments"][1:]:
            assert e["scenario_fingerprint"] == fp
        assert fp["frames_count"] == 16
        assert fp["objects_count"] == 8
        assert study["scenario"]["seed"] == 42

    def test_frames_selected_from_candidates(self, study):
        for e in study["experiments"]:
            for frame_id in e["frames_selected"]:
                assert frame_id == "" or (frame_id.startswith("FRM-") and len(frame_id) == 8)


class TestReproducibility:
    def test_same_params_same_result(self, study):
        params = ConvergenceParams(experiments=4, iterations=8, seed_start=100)
        again = run_convergence_study(make_config(), params, workers=1)
        for a, b in zip(study["experiments"], again["experiments"]):
            assert a["trajectory"] == b["trajectory"]
            assert a["frames_selected"] == b["frames_selected"]

    def test_different_seed_start_changes_trajectories(self):
        cfg = make_config()
        s1 = run_convergence_study(cfg, ConvergenceParams(experiments=3, iterations=6, seed_start=0), workers=1)
        s2 = run_convergence_study(cfg, ConvergenceParams(experiments=3, iterations=6, seed_start=500), workers=1)
        t1 = [tuple(e["trajectory"]) for e in s1["experiments"]]
        t2 = [tuple(e["trajectory"]) for e in s2["experiments"]]
        assert t1 != t2
        # район при этом не перегенерирован
        assert s1["experiments"][0]["scenario_fingerprint"] == s2["experiments"][0]["scenario_fingerprint"]

    def test_parallel_matches_sequential(self):
        cfg = make_config()
        params = ConvergenceParams(experiments=4, iterations=6, seed_start=7)
        seq = run_convergence_study(cfg, params, workers=1)
        par = run_convergence_study(cfg, params, workers=2)
        for a, b in zip(seq["experiments"], par["experiments"]):
            assert a["trajectory"] == b["trajectory"]


class TestIterationStats:
    def test_known_values(self):
        trajectories = [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.4],
            [0.3, 0.3, 0.5],
            [0.4, 0.6, 0.6],
        ]
        stats = compute_iteration_stats(trajectories)
        assert len(stats) == 3
        arr = np.asarray(trajectories, dtype=float)
        for k, s in enumerate(stats):
            col = arr[:, k]
            assert s["iteration"] == k + 1
            assert s["mean"] == pytest.approx(col.mean())
            assert s["std"] == pytest.approx(col.std(ddof=1))
            assert s["median"] == pytest.approx(np.median(col))
            assert s["p10"] == pytest.approx(np.percentile(col, 10))
            assert s["p90"] == pytest.approx(np.percentile(col, 90))
            # 95%-й ДИ среднего: t(3) = 3.182 при M = 4
            half = 3.182 * col.std(ddof=1) / math.sqrt(4)
            assert s["ci95_low"] == pytest.approx(col.mean() - half, rel=1e-3)
            assert s["ci95_high"] == pytest.approx(col.mean() + half, rel=1e-3)
            assert s["ci95_low"] <= s["mean"] <= s["ci95_high"]
            prev = arr[:, k - 1] if k > 0 else np.zeros(4)
            assert s["mean_increment"] == pytest.approx((col - prev).mean())
        # первый прирост — от нулевого начального покрытия
        assert stats[0]["mean_increment"] == pytest.approx(arr[:, 0].mean())

    def test_single_experiment_gives_nan_std_and_ci(self):
        stats = compute_iteration_stats([[0.5, 0.6]])
        assert len(stats) == 2
        assert math.isnan(stats[0]["std"])
        assert math.isnan(stats[0]["ci95_low"])
        assert math.isnan(stats[0]["ci95_high"])
        assert stats[0]["mean"] == pytest.approx(0.5)


def _make_stats(means, increments, std=0.001):
    return [
        {
            "iteration": i + 1, "mean": m, "std": std, "median": m,
            "p10": m - std, "p90": m + std,
            "ci95_low": m - std, "ci95_high": m + std,
            "mean_increment": inc,
        }
        for i, (m, inc) in enumerate(zip(means, increments))
    ]


class TestDiagnoseConvergence:
    def test_converged(self):
        stats = _make_stats(
            means=[0.4, 0.6, 0.8, 0.9, 0.92, 0.93],
            increments=[0.4, 0.2, 0.05, 0.002, 0.001, 0.0005],
        )
        params = ConvergenceParams(stabilization_tol=1e-3, stabilization_window=2, min_completeness=0.9)
        d = diagnose_convergence(stats, params)
        assert d["verdict"] == "converged"
        assert d["stabilized"] and d["stabilized_at_iteration"] == 6
        assert not d["local_optima_detected"]

    def test_stagnation_below_required(self):
        stats = _make_stats(
            means=[0.1, 0.2, 0.3, 0.31, 0.312, 0.313],
            increments=[0.1, 0.1, 0.01, 0.002, 0.001, 0.0005],
        )
        params = ConvergenceParams(stabilization_tol=1e-3, stabilization_window=2, min_completeness=0.9)
        d = diagnose_convergence(stats, params)
        assert d["verdict"] == "stagnation"
        assert d["stabilized"]

    def test_no_convergence(self):
        stats = _make_stats(
            means=[0.2, 0.4, 0.6, 0.8],
            increments=[0.2, 0.2, 0.2, 0.2],
        )
        params = ConvergenceParams(stabilization_tol=1e-3, stabilization_window=2, min_completeness=0.9)
        d = diagnose_convergence(stats, params)
        assert d["verdict"] == "no_convergence"
        assert not d["stabilized"]
        assert d["stabilized_at_iteration"] is None

    def test_different_local_optima(self):
        stats = _make_stats(
            means=[0.4, 0.6, 0.8, 0.85, 0.86, 0.861],
            increments=[0.4, 0.2, 0.05, 0.002, 0.001, 0.0005],
            std=0.05,  # итоговый разброс между экспериментами выше порога
        )
        params = ConvergenceParams(stabilization_tol=1e-3, stabilization_window=2, min_completeness=0.9,
                                   spread_tol=0.02)
        d = diagnose_convergence(stats, params)
        assert d["verdict"] == "different_local_optima"
        assert d["local_optima_detected"]

    def test_diagnose_requires_nonempty_stats(self):
        with pytest.raises(ValueError):
            diagnose_convergence([], ConvergenceParams())


class TestParamsValidation:
    def test_bad_experiments(self):
        with pytest.raises(ValueError):
            ConvergenceParams(experiments=1)

    def test_bad_iterations(self):
        with pytest.raises(ValueError):
            ConvergenceParams(iterations=0)

    def test_bad_tolerance_and_window(self):
        with pytest.raises(ValueError):
            ConvergenceParams(stabilization_tol=0.0)
        with pytest.raises(ValueError):
            ConvergenceParams(stabilization_window=0)

    def test_bad_temperature(self):
        with pytest.raises(ValueError):
            ConvergenceParams(temperature=0.0)

    def test_bad_min_completeness(self):
        with pytest.raises(ValueError):
            ConvergenceParams(min_completeness=1.5)

    def test_frames_required(self):
        cfg = make_config(frames__enabled=False)
        with pytest.raises(ConfigError):
            run_convergence_study(cfg, ConvergenceParams(experiments=2, iterations=2), workers=1)


class TestExports:
    def test_csv_json_png_written(self, study, tmp_path):
        csv_paths = write_convergence_csv(study, tmp_path)
        for name in ("convergence_trajectories.csv", "convergence_summary.csv", "convergence_experiments.csv"):
            assert name in csv_paths and csv_paths[name].exists()

        traj_lines = csv_paths["convergence_trajectories.csv"].read_text(encoding="utf-8").splitlines()
        assert traj_lines[0].startswith("experiment,seed,iteration,info_completeness")
        assert len(traj_lines) == 1 + 4 * 8  # заголовок + M×K строк

        summary_lines = csv_paths["convergence_summary.csv"].read_text(encoding="utf-8").splitlines()
        assert summary_lines[0] == "iteration,mean,std,median,p10,p90,ci95_low,ci95_high,mean_increment"
        assert len(summary_lines) == 1 + 8

        json_path = write_convergence_json(study, tmp_path)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["params"]["experiments"] == 4
        assert loaded["convergence"]["verdict"] == study["convergence"]["verdict"]
        assert len(loaded["experiments"][0]["trajectory"]) == 8

        chart_paths = render_convergence_charts(study, tmp_path)
        for name in ("convergence_mean_spread.png", "convergence_std.png",
                     "convergence_trajectories.png", "convergence_increments.png"):
            assert name in chart_paths
            assert chart_paths[name].stat().st_size > 0


class TestSelectionModes:
    def test_greedy_is_deterministic(self):
        cfg = make_config()
        params = ConvergenceParams(experiments=2, iterations=6, seed_start=100, selection="greedy")
        a = run_experiment(cfg, params, algo_seed=1, experiment_index=0)
        b = run_experiment(cfg, params, algo_seed=999, experiment_index=1)
        assert a["trajectory"] == b["trajectory"]

    def test_random_differs_across_seeds(self):
        cfg = make_config()
        params = ConvergenceParams(experiments=2, iterations=6, seed_start=100, selection="random")
        a = run_experiment(cfg, params, algo_seed=1, experiment_index=0)
        b = run_experiment(cfg, params, algo_seed=2, experiment_index=1)
        assert a["trajectory"] != b["trajectory"]
        # приросты во всех режимах — фактическая разность траектории
        for k in range(1, 6):
            assert abs(a["marginal_gains"][k] - (a["trajectory"][k] - a["trajectory"][k - 1])) < 1e-12

    def test_greedy_not_worse_than_random_on_average(self):
        cfg = make_config()
        final = {}
        for selection in ("greedy", "random"):
            study = run_convergence_study(
                cfg, ConvergenceParams(experiments=6, iterations=8, seed_start=100, selection=selection),
                workers=1,
            )
            final[selection] = study["iteration_stats"][-1]["mean"]
        assert final["greedy"] >= final["random"]

    def test_invalid_selection(self):
        with pytest.raises(ValueError):
            ConvergenceParams(selection="genetic")


class TestThresholdAnalysis:
    def test_known_k_confidence(self):
        # 4 эксперимента: порог 0.5 достигается к k=4 у всех, к k=2 — у двоих
        trajectories = [
            [0.1, 0.3, 0.6, 0.7],
            [0.1, 0.55, 0.6, 0.7],
            [0.2, 0.3, 0.55, 0.7],
            [0.2, 0.6, 0.65, 0.7],
        ]
        res = coverage_threshold_analysis(trajectories, thresholds=(0.5,), confidence=0.95)
        assert len(res) == 1
        r = res[0]
        assert r["k_confidence"] == 3          # доли: k=1 → 0, k=2 → 0.5, k=3 → 1.0 ≥ 0.95
        assert r["reach_share_final"] == 1.0
        assert r["k_median_first_reach"] == 2.5  # первые попадания: 3, 2, 3, 2
        assert r["experiments_reached"] == 4
        assert len(r["share_by_iteration"]) == 4

    def test_unreachable_threshold(self):
        trajectories = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        r = coverage_threshold_analysis(trajectories, thresholds=(0.9,), confidence=0.95)[0]
        assert r["k_confidence"] is None
        assert r["k_median_first_reach"] is None
        assert r["experiments_reached"] == 0
        assert r["reach_share_final"] == 0.0

    def test_ci_contains_share(self, study):
        for r in coverage_threshold_analysis(study, thresholds=(0.5, 0.9)):
            assert r["reach_share_final_ci95_low"] <= r["reach_share_final"] <= r["reach_share_final_ci95_high"]
            assert 0.0 <= r["reach_share_final_ci95_low"] <= 1.0

    def test_bad_confidence(self):
        with pytest.raises(ValueError):
            coverage_threshold_analysis([[0.1]], confidence=1.5)


class TestStdTrendAnalysis:
    def _stats_with_std(self, stds):
        return [
            {
                "iteration": i + 1, "mean": 0.5, "std": s, "median": 0.5,
                "p10": 0.5 - s, "p90": 0.5 + s,
                "ci95_low": 0.4, "ci95_high": 0.6, "mean_increment": 0.01,
            }
            for i, s in enumerate(stds)
        ]

    def test_decreasing_trend_detected(self):
        stds = [0.10, 0.09, 0.085, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
                0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0005, 0.0002]
        res = std_trend_analysis(self._stats_with_std(stds), n_perm=2000)
        assert res["spearman_rho_std"] < -0.9
        assert res["decreasing_significant"]
        assert res["std_ratio_last_first"] < 0.5

    def test_constant_std_is_no_trend(self):
        res = std_trend_analysis(self._stats_with_std([0.01] * 12), n_perm=500)
        assert res["spearman_rho_std"] == 0.0
        assert not res["decreasing_significant"]
        assert res["permutation_p_value"] == 1.0

    def test_greedy_study_zero_std(self):
        cfg = make_config()
        study = run_convergence_study(
            cfg, ConvergenceParams(experiments=2, iterations=6, selection="greedy"), workers=1,
        )
        assert all(s["std"] == 0.0 for s in study["iteration_stats"])
        res = std_trend_analysis(study["iteration_stats"], n_perm=500)
        assert not res["decreasing_significant"]


class TestComparisonCharts:
    def test_comparison_charts_written(self, study, tmp_path):
        other = run_convergence_study(
            make_config(),
            ConvergenceParams(experiments=4, iterations=8, seed_start=100, selection="random"),
            workers=1,
        )
        paths = render_comparison_charts([("softmax T=0.25", study), ("random", other)], tmp_path)
        assert set(paths) == {"convergence_comparison_curves.png", "convergence_comparison_summary.png"}
        for p in paths.values():
            assert p.stat().st_size > 0
