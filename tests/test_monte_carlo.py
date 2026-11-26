"""Тесты метода Монте-Карло: сходимости средней геометрической полноты покрытия района."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.monte_carlo import (
    METRIC_GEOMETRIC,
    METRIC_HIDDEN,
    MonteCarloParams,
    _greedy_select,
    compute_running_stats,
    convergence_assessment,
    render_monte_carlo_charts,
    required_experiments_for_precision,
    run_monte_carlo_experiment,
    run_monte_carlo_study,
    write_monte_carlo_csv,
    write_monte_carlo_json,
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
def params() -> MonteCarloParams:
    return MonteCarloParams(
        experiments=12,
        iterations=6,
        checkpoints=(4, 8, 12),
        seed_start=100,
        cloud_fraction=0.5,
        selection="greedy",
    )


@pytest.fixture(scope="module")
def study(params):
    return run_monte_carlo_study(make_config(), params, workers=1)


class TestParamsValidation:
    def test_defaults(self):
        p = MonteCarloParams()
        assert p.experiments == 1000 and p.iterations == 70
        assert p.checkpoints[-1] == 1000

    def test_experiments_at_least_two(self):
        with pytest.raises(ValueError, match="M должно быть не меньше 2"):
            MonteCarloParams(experiments=1)

    def test_checkpoints_sorted_and_within_m(self):
        with pytest.raises(ValueError, match="возрастающими"):
            MonteCarloParams(checkpoints=(10, 5))
        with pytest.raises(ValueError, match="не может превосходить"):
            MonteCarloParams(experiments=100, checkpoints=(10, 200))

    def test_cloud_fraction_range(self):
        with pytest.raises(ValueError, match="cloud_fraction"):
            MonteCarloParams(cloud_fraction=1.5)

    def test_selection_values(self):
        with pytest.raises(ValueError, match="selection"):
            MonteCarloParams(selection="unknown")

    def test_precision_positive(self):
        with pytest.raises(ValueError, match="precision"):
            MonteCarloParams(precision=0.0)


class TestSingleExperiment:
    def test_deterministic_by_seed(self, params):
        a = run_monte_carlo_experiment(make_config(), params, 7, 0)
        b = run_monte_carlo_experiment(make_config(), params, 7, 0)
        assert a["final_geometric_completeness"] == b["final_geometric_completeness"]
        assert a["hidden_objects_coverage"] == b["hidden_objects_coverage"]
        assert a["frames_selected"] == b["frames_selected"]

    def test_different_seeds_differ(self, params):
        a = run_monte_carlo_experiment(make_config(), params, 7, 0)
        b = run_monte_carlo_experiment(make_config(), params, 8, 0)
        assert a["seed"] != b["seed"]
        # сценарии разные: хотя бы одна характеристика отличается
        assert (a["n_hidden"], a["optical_district_coverage"]) != (
            b["n_hidden"], b["optical_district_coverage"],
        )

    def test_metrics_bounded(self, params):
        e = run_monte_carlo_experiment(make_config(), params, 11, 0)
        assert 0.0 <= e["final_geometric_completeness"] <= 1.0
        assert math.isnan(e["hidden_objects_coverage"]) or 0.0 <= e["hidden_objects_coverage"] <= 1.0
        assert e["n_hidden"] <= e["n_objects"]
        assert len(e["frames_selected"]) <= params.iterations

    def test_plan_not_executed_note(self, params):
        # запланированные кадры — план; в данных эксперимента нет признаков «выполненной» съёмки
        e = run_monte_carlo_experiment(make_config(), params, 5, 0)
        assert isinstance(e["frames_selected"], list)
        assert all(f.startswith("FRM-") for f in e["frames_selected"])

    def test_no_hidden_objects(self):
        # cloud_fraction=0: скрытых объектов нет, метрика C не определена (NaN),
        # итоговая полнота — оптическая + досъёмка района
        p = MonteCarloParams(experiments=2, iterations=4, checkpoints=(2,),
                             cloud_fraction=0.0, selection="greedy")
        e = run_monte_carlo_experiment(make_config(), p, 3, 0)
        assert e["n_hidden"] == 0
        assert math.isnan(e["hidden_objects_coverage"])
        assert 0.0 < e["final_geometric_completeness"] <= 1.0


class TestLazyGreedy:
    def test_lazy_equals_naive_greedy(self, params):
        # ленивый жадный выбор должен совпадать с наивным перебором
        from shapely.geometry import GeometryCollection
        from shapely.ops import unary_union

        from sro_generator.generator import generate_scenario

        scenario = generate_scenario(make_config())
        district = scenario.district_local
        district_area = float(district.area)
        rng = np.random.default_rng(0)
        hidden = [o.geometry for o in scenario.objects if rng.random() < 0.5]
        hidden_union = unary_union(hidden) if hidden else None
        hidden_area = float(hidden_union.area) if hidden_union is not None else 0.0

        cand_d = [f.geometry.intersection(district) for f in scenario.frames]
        area_d = [g.area for g in cand_d]
        if hidden_union is not None:
            cand_h = [f.geometry.intersection(hidden_union) for f in scenario.frames]
            area_h = [g.area for g in cand_h]
        else:
            cand_h, area_h = None, None

        from sro_generator.monte_carlo import _combined_gain

        # наивный жадный
        remaining = list(range(len(scenario.frames)))
        union_d = GeometryCollection()
        union_h = GeometryCollection()
        naive: list[int] = []
        for _ in range(min(6, len(remaining))):
            gains = [
                _combined_gain(i, cand_d, cand_h, area_d, area_h, union_d, union_h,
                               district_area, hidden_area)
                for i in remaining
            ]
            idx = remaining[int(np.argmax(gains))]
            naive.append(idx)
            remaining.remove(idx)
            union_d = cand_d[idx] if union_d.is_empty else union_d.union(cand_d[idx])
            if cand_h is not None:
                union_h = cand_h[idx] if union_h.is_empty else union_h.union(cand_h[idx])

        lazy = _greedy_select(
            len(scenario.frames),
            MonteCarloParams(experiments=2, iterations=6, checkpoints=(2,),
                             selection="greedy"),
            np.random.default_rng(0),
            cand_d, cand_h, area_d, area_h, district_area, hidden_area,
        )[:6]
        assert lazy == naive

    def test_random_and_softmax_lengths(self):
        from sro_generator.generator import generate_scenario

        scenario = generate_scenario(make_config())
        n = len(scenario.frames)
        cand_d = [f.geometry for f in scenario.frames]
        for selection in ("random", "softmax"):
            p = MonteCarloParams(experiments=2, iterations=5, checkpoints=(2,),
                                 selection=selection)
            sel = _greedy_select(n, p, np.random.default_rng(1), cand_d, None,
                                 [g.area for g in cand_d], None, 1.0, 0.0)
            assert len(sel) == 5
            assert len(set(sel)) == 5


class TestRunningStats:
    def test_checkpoint_math(self):
        values = [0.5, 0.7, 0.9, 0.6]
        rows = compute_running_stats(values, (2, 4))
        assert rows[0]["m"] == 2
        assert rows[0]["mean"] == pytest.approx(0.6)
        assert rows[0]["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
        assert rows[0]["sem"] == pytest.approx(rows[0]["std"] / math.sqrt(2))
        assert rows[0]["ci95_half"] == pytest.approx(12.706 * rows[0]["sem"])
        assert rows[1]["mean"] == pytest.approx(0.675)

    def test_nan_excluded(self):
        rows = compute_running_stats([0.5, math.nan, 0.9, 0.6], (4,))
        assert rows[0]["n_valid"] == 3
        assert rows[0]["mean"] == pytest.approx(np.nanmean([0.5, math.nan, 0.9, 0.6]))

    def test_checkpoint_beyond_samples(self):
        with pytest.raises(ValueError, match="превышает"):
            compute_running_stats([0.1, 0.2], (3,))

    def test_required_experiments_formula(self):
        # s=0.05, δ=0.01, z=1.96 -> n* = (1.96*5)^2 = 96.04 -> 97
        values = [0.5] * 50 + [0.6] * 50  # std ≈ 0.05025
        res = required_experiments_for_precision(values, precision=0.01, confidence=0.95)
        assert res["z"] == pytest.approx(1.96)
        assert res["std"] == pytest.approx(0.050251, abs=1e-5)
        assert res["n_required"] == pytest.approx(math.ceil((1.96 * res["std"] / 0.01) ** 2))
        assert res["sufficient"] == (res["n_sample"] >= res["n_required"])

    def test_required_experiments_validation(self):
        with pytest.raises(ValueError, match="точность"):
            required_experiments_for_precision([0.1, 0.2], precision=0.0)


class TestStudy:
    def test_structure(self, study, params):
        assert study["mode"] == "monte_carlo_mean_convergence"
        assert len(study["experiments"]) == params.experiments
        for metric in (METRIC_GEOMETRIC, METRIC_HIDDEN):
            rows = study["running_stats"][metric]
            assert [r["m"] for r in rows] == [4, 8, 12]
            assert len(study["required_experiments"][metric]) > 0
            assert metric in study["assessment"]

    def test_scenarios_differ_between_runs(self, study):
        keys = {(e["n_objects"], e["n_hidden"], round(e["optical_district_coverage"], 9))
                for e in study["experiments"]}
        assert len(keys) > 1  # сценарии различаются между запусками

    def test_assessment_flags(self, study):
        for metric in (METRIC_GEOMETRIC, METRIC_HIDDEN):
            a = study["assessment"][metric]
            assert set(a) >= {"mean_final", "ci95_half_final", "mean_drift_last",
                              "ci_half_loglog_slope", "mean_converged",
                              "uncertainty_decreases", "precision_reached"}
            assert isinstance(a["mean_converged"], bool)
            assert isinstance(a["uncertainty_decreases"], bool)

    def test_progress_callback(self, params):
        seen: list[int] = []
        run_monte_carlo_study(
            make_config(), params, workers=1,
            progress=lambda done, total: seen.append(done),
        )
        assert seen == list(range(1, params.experiments + 1))

    def test_frames_disabled_rejected(self, params):
        cfg = make_config(frames__enabled=False)
        with pytest.raises(ConfigError, match="frames"):
            run_monte_carlo_study(cfg, params, workers=1)

    def test_workers_do_not_change_results(self, params):
        s1 = run_monte_carlo_study(make_config(), params, workers=1)
        s2 = run_monte_carlo_study(make_config(), params, workers=2)
        v1 = [e["final_geometric_completeness"] for e in s1["experiments"]]
        v2 = [e["final_geometric_completeness"] for e in s2["experiments"]]
        assert v1 == v2


class TestExport:
    def test_csv(self, study, tmp_path):
        paths = write_monte_carlo_csv(study, tmp_path)
        assert (tmp_path / "monte_carlo_experiments.csv").exists()
        assert (tmp_path / "monte_carlo_checkpoints.csv").exists()
        text = (tmp_path / "monte_carlo_checkpoints.csv").read_text(encoding="utf-8")
        assert "final_geometric_completeness" in text and "hidden_objects_coverage" in text

    def test_json(self, study, tmp_path):
        p = write_monte_carlo_json(study, tmp_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["mode"] == "monte_carlo_mean_convergence"
        assert len(data["experiments"]) == len(study["experiments"])

    def test_charts(self, study, tmp_path):
        paths = render_monte_carlo_charts(study, tmp_path)
        assert len(paths) == 4
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0

    def test_charts_constant_values(self, tmp_path):
        # почти константные значения (реально для C при большом K) не должны ронять гистограмму
        study = {
            "params": {"precision": 0.01, "iterations": 70, "selection": "greedy",
                       "temperature": 0.25, "cloud_fraction": 0.4},
            "experiments": [
                {"hidden_objects_coverage": 1.0, "final_geometric_completeness": 0.5 + i * 1e-9}
                for i in range(8)
            ],
            "running_stats": {
                METRIC_HIDDEN: [
                    {"m": 4, "mean": 1.0, "std": 0.0, "sem": 0.0, "ci95_low": 1.0,
                     "ci95_high": 1.0, "ci95_half": 0.0},
                    {"m": 8, "mean": 1.0, "std": 0.0, "sem": 0.0, "ci95_low": 1.0,
                     "ci95_high": 1.0, "ci95_half": 0.0},
                ],
                METRIC_GEOMETRIC: [
                    {"m": 4, "mean": 0.5, "std": 1e-9, "sem": 5e-10, "ci95_low": 0.5,
                     "ci95_high": 0.5, "ci95_half": 1e-9},
                    {"m": 8, "mean": 0.5, "std": 1e-9, "sem": 5e-10, "ci95_low": 0.5,
                     "ci95_high": 0.5, "ci95_half": 1e-9},
                ],
            },
            "assessment": {
                METRIC_HIDDEN: {"ci_half_loglog_slope": math.nan, "precision_reached": True},
                METRIC_GEOMETRIC: {"ci_half_loglog_slope": -0.5, "precision_reached": True},
            },
        }
        paths = render_monte_carlo_charts(study, tmp_path)
        assert len(paths) == 4
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0


class TestCloudSensitivity:
    @pytest.fixture(scope="module")
    def grid(self):
        from sro_generator.monte_carlo import run_cloud_sensitivity

        return run_cloud_sensitivity(
            make_config(),
            cloud_fractions=(0.2, 0.6),
            k_values=(5, 10),
            selections=("greedy", "softmax"),
            experiments=6,
            seed_start=50,
            workers=1,
        )

    def test_grid_structure(self, grid):
        assert len(grid["cells"]) == 2 * 2 * 2
        for cell in grid["cells"]:
            assert len(cell["experiments"]) == 6
            assert 0.0 <= cell["cloud_fraction"] <= 1.0
            for e in cell["experiments"]:
                assert "objects_area_fraction" in e
                assert "hidden_area_fraction" in e
                assert "n_candidates" in e

    def test_paired_seeds_across_cells(self, grid):
        seeds0 = {e["seed"] for e in grid["cells"][0]["experiments"]}
        for cell in grid["cells"][1:]:
            assert {e["seed"] for e in cell["experiments"]} == seeds0

    def test_combo_stats(self, grid):
        from sro_generator.monte_carlo import compute_combo_stats

        s = compute_combo_stats(grid["cells"][0]["experiments"], thresholds=(0.9, 0.99))
        assert s["n_runs"] == 6
        assert 0.0 <= s["g"]["mean"] <= 1.0
        assert s["g"]["ci95_half"] > 0
        assert set(s["thresholds"]) == {0.9, 0.99}
        for gamma in (0.9, 0.99):
            t = s["thresholds"][gamma]["g"]
            assert 0.0 <= t["share"] <= 1.0
            assert t["ci95_low"] <= t["share"] <= t["ci95_high"]

    def test_min_k(self, grid):
        from sro_generator.monte_carlo import compute_combo_stats, min_k_for_thresholds

        for cell in grid["cells"]:
            cell["stats"] = compute_combo_stats(cell["experiments"])
        rows = min_k_for_thresholds(grid, thresholds=(0.5,), confidence=0.5)
        assert len(rows) == 2 * 2  # 2 облачности × 2 правила
        for row in rows:
            assert row["threshold"] == 0.5
            assert row["k_min"] in (5, 10, None)

    def test_verify_cloud_model(self, grid):
        from sro_generator.monte_carlo import verify_cloud_model

        checks = verify_cloud_model(grid)
        for c in checks.values():
            assert c["ok"]
        # оптическое покрытие = площадь объектов − площадь скрытых
        assert checks["optical_equals_objects_minus_hidden"]["max_abs_diff"] < 1e-9

    def test_workers_invariance(self, grid):
        from sro_generator.monte_carlo import run_cloud_sensitivity

        g2 = run_cloud_sensitivity(
            make_config(), cloud_fractions=(0.4,), k_values=(5,),
            selections=("greedy",), experiments=4, seed_start=50, workers=2,
        )
        v1 = [e["final_geometric_completeness"] for e in grid["cells"][0]["experiments"]][:4]
        v2 = [e["final_geometric_completeness"] for e in g2["cells"][0]["experiments"]]
        # разные облачности/сценарии — сравниваем только воспроизводимость workers=2
        g3 = run_cloud_sensitivity(
            make_config(), cloud_fractions=(0.4,), k_values=(5,),
            selections=("greedy",), experiments=4, seed_start=50, workers=1,
        )
        v3 = [e["final_geometric_completeness"] for e in g3["cells"][0]["experiments"]]
        assert v2 == v3

    def test_csv_json_charts(self, grid, tmp_path):
        from sro_generator.monte_carlo import (
            compute_combo_stats, render_sensitivity_charts,
            write_sensitivity_csv, write_sensitivity_json,
        )

        for cell in grid["cells"]:
            cell["stats"] = compute_combo_stats(cell["experiments"])
        paths = write_sensitivity_csv(grid, tmp_path)
        assert (tmp_path / "cloud_sensitivity_summary.csv").exists()
        assert (tmp_path / "cloud_sensitivity_thresholds.csv").exists()
        p = write_sensitivity_json(grid, tmp_path)
        import json as _json
        data = _json.loads(p.read_text(encoding="utf-8"))
        assert data["mode"] == "cloud_coverage_sensitivity"
        charts = render_sensitivity_charts(grid, tmp_path)
        assert len(charts) == 4
        for c in charts.values():
            assert c.exists() and c.stat().st_size > 0


class TestClassicalStats:
    def test_t_crit_tables(self):
        from sro_generator.monte_carlo import t_crit

        assert t_crit(0.90, 1) == pytest.approx(6.314)
        assert t_crit(0.95, 1) == pytest.approx(12.706)
        assert t_crit(0.99, 1) == pytest.approx(63.657)
        assert t_crit(0.95, 30) == pytest.approx(2.042)
        assert t_crit(0.99, 30) == pytest.approx(2.750)
        assert t_crit(0.95, 500) == pytest.approx(1.96)  # нормальное приближение
        # прочие β — нормальное приближение z(β) без ошибки
        assert t_crit(0.93, 5) > 1.6

    def test_classical_series_math(self):
        from sro_generator.monte_carlo import classical_series

        values = [0.5, 0.7, 0.9, 0.6, 0.8]
        s = classical_series(values, beta=0.95)
        assert len(s) == 4  # n = 2..5
        last = s[-1]
        assert last["n"] == 5
        assert last["m_star"] == pytest.approx(0.7)
        assert last["d_tilde"] == pytest.approx(0.025)
        assert last["sigma_tilde"] == pytest.approx(math.sqrt(0.025))
        assert last["epsilon"] == pytest.approx(2.776 * last["sigma_tilde"] / math.sqrt(5))
        assert last["ci_low"] == pytest.approx(last["m_star"] - last["epsilon"])
        # точность убывает с ростом n
        assert s[-1]["epsilon"] < s[0]["epsilon"]

    def test_classical_series_nan_and_small(self):
        from sro_generator.monte_carlo import classical_series

        s = classical_series([0.5, math.nan, 0.7, 0.9, 0.6], beta=0.90)
        assert s[-1]["n"] == 4
        with pytest.raises(ValueError):
            classical_series([0.5, 0.6], beta=0.95)
        with pytest.raises(ValueError):
            classical_series([0.5, 0.6, 0.7], beta=0.93)

    def test_required_trials(self):
        from sro_generator.monte_carlo import required_trials

        r = required_trials(0.05, 0.01, beta=0.95)
        assert r["n_required"] == 97  # (1.96*5)^2 = 96.04
        assert r["normal_approximation"] is True
        r99 = required_trials(0.05, 0.01, beta=0.99)
        assert r99["n_required"] == math.ceil((2.576 * 5) ** 2)
        with pytest.raises(ValueError):
            required_trials(0.05, 0.0)
        with pytest.raises(ValueError):
            required_trials(-0.1, 0.01)

    def test_k_sweep_and_charts(self, tmp_path):
        from sro_generator.monte_carlo import (
            classical_series, k_sweep, render_classical_charts, required_trials,
        )

        sweep = k_sweep(make_config(), cloud_fraction=0.4, k_values=(5, 10),
                        selection="greedy", experiments=6, seed_start=70,
                        beta=0.95, workers=1, keep_runs=True)
        assert set(sweep) == {5, 10}
        for k, row in sweep.items():
            assert row["g"]["n"] == 6
            assert 0.0 <= row["g"]["mean"] <= 1.0
            assert row["g"]["ci_half"] > 0
            assert len(row["runs"]) == 6
        # накопленные оценки по испытаниям при фиксированном K
        series = classical_series([r["final_geometric_completeness"] for r in sweep[10]["runs"]], 0.95)
        assert series[-1]["m_star"] == pytest.approx(sweep[10]["g"]["mean"])
        n_req = required_trials(series[-1]["sigma_tilde"], 0.01, 0.95)
        assert n_req["n_required"] >= 3
        paths = render_classical_charts(series, sweep, 0.95, 0.01, tmp_path)
        assert len(paths) == 5
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0

    def test_k_sweep_workers_invariant(self):
        from sro_generator.monte_carlo import k_sweep

        s1 = k_sweep(make_config(), 0.4, (5,), "greedy", 4, seed_start=70, workers=1)
        s2 = k_sweep(make_config(), 0.4, (5,), "greedy", 4, seed_start=70, workers=2)
        assert s1[5]["g"]["mean"] == s2[5]["g"]["mean"]
