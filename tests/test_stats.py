"""Тесты серии параллельных прогонов и статистики."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sro_generator.config import ScenarioConfig
from sro_generator.stats import (
    QUADRANT_GRID,
    chi2_uniform,
    render_report_charts,
    run_batch,
    run_single,
    summarize,
    write_batch_csv,
)

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


@pytest.fixture(scope="module")
def batch():
    return run_batch(make_config(), n_runs=6, workers=2, seed_start=100)


class TestRunBatch:
    def test_all_runs_completed(self, batch):
        assert len(batch["runs"]) == 6
        assert batch["summary"]["n_runs"] == 6

    def test_all_runs_ok(self, batch):
        assert batch["summary"]["n_ok"] == 6
        assert batch["summary"]["n_failed"] == 0
        assert all(r["ok"] for r in batch["runs"])

    def test_seeds_assigned_in_order(self, batch):
        assert [r["seed"] for r in batch["runs"]] == [100, 101, 102, 103, 104, 105]

    def test_runs_sorted_by_seed(self, batch):
        seeds = [r["seed"] for r in batch["runs"]]
        assert seeds == sorted(seeds)

    def test_progress_callback_called(self):
        calls = []
        run_batch(make_config(objects__count=5), n_runs=4, workers=2, seed_start=0,
                  progress=lambda done, total: calls.append((done, total)))
        assert calls and calls[-1] == (4, 4)

    def test_runs_differ(self, batch):
        coords = [[(o["cx_m"], o["cy_m"]) for o in r["objects"]] for r in batch["runs"]]
        assert len({tuple(c) for c in coords}) == len(coords)

    def test_sequential_matches_parallel(self, batch):
        seq = run_batch(make_config(), n_runs=6, workers=1, seed_start=100)
        par_areas = [[o["area_m2"] for o in r["objects"]] for r in batch["runs"]]
        seq_areas = [[o["area_m2"] for o in r["objects"]] for r in seq["runs"]]
        assert par_areas == seq_areas


class TestRunStats:
    def test_per_run_structure(self, batch):
        for r in batch["runs"]:
            assert len(r["objects"]) == 25
            assert r["attempts_total"] > 0
            assert r["duration_s"] > 0
            assert 0.0 < r["fill_ratio"] < 1.0
            assert len(r["quadrant_counts"]) == QUADRANT_GRID * QUADRANT_GRID
            assert sum(r["quadrant_counts"]) == 25
            assert r["nnd_mean_m"] >= r["nnd_min_m"] > 0

    def test_object_fields_in_range(self, batch):
        for r in batch["runs"]:
            for o in r["objects"]:
                assert 500_000 * 0.99 <= o["area_m2"] <= 5_000_000 * 1.01
                assert 3 <= o["vertex_count"] <= 10
                assert o["class_label"] in ("warehouse", "industrial", "transport")
                assert 1 <= o["priority"] <= 5
                assert o["deadline_h"] in (30.0, 48.0)
                assert o["resolution_m"] in (0.5, 1.0)

    def test_coverage_in_unit_interval(self, batch):
        for r in batch["runs"]:
            for c in r["coverage"]:
                assert 0.0 <= c <= 1.0
            assert r["full_coverage_frames"] <= len(r["coverage"])

    def test_failed_run_captured_not_raised(self):
        cfg = make_config(
            district__width_km=5.0,
            district__height_km=5.0,
            objects__count=200,
            objects__area_max_m2=5_000_000.0,
            placement__max_attempts_per_object=20,
        )
        result = run_batch(cfg, n_runs=3, workers=1, seed_start=7)
        assert result["summary"]["n_failed"] == 3
        assert result["summary"]["n_ok"] == 0
        for r in result["runs"]:
            assert not r["ok"]
            assert "OBJ-" in r["error"]

    def test_partial_failure(self):
        cfg = make_config(
            district__width_km=5.0,
            district__height_km=5.0,
            objects__count=200,
            objects__area_max_m2=5_000_000.0,
            placement__max_attempts_per_object=20,
        )
        result = run_batch(cfg, n_runs=2, workers=1, seed_start=1)
        summary = summarize(
            result["runs"] + [{"seed": 999, "ok": True, "error": None, "duration_s": 0.1,
                               "attempts_total": 100, "warnings": [], "district_area_m2": 1e8,
                               "objects_total_area_m2": 1e6, "fill_ratio": 0.01,
                               "nnd_mean_m": 500.0, "nnd_min_m": 100.0,
                               "quadrant_counts": [3] * 9, "coverage": [1.0, 0.5],
                               "full_coverage_frames": 1, "objects": []}],
            cfg.to_dict(),
        )
        assert summary["n_runs"] == 3
        assert summary["n_failed"] == 2
        assert summary["n_ok"] == 1
        assert summary["errors"] and summary["errors"][0]["seed"] == 1


class TestSummarize:
    def test_summary_values(self, batch):
        s = batch["summary"]
        assert s["attempts_total"]["mean"] > 0
        assert s["attempts_total"]["min"] <= s["attempts_total"]["mean"] <= s["attempts_total"]["max"]
        assert 0.0 < s["fill_ratio"]["mean"] < 1.0
        assert s["areas_m2"]["min"] >= 500_000 * 0.99
        assert s["areas_m2"]["max"] <= 5_000_000 * 1.01
        assert sum(s["vertex_counts"].values()) == 6 * 25
        assert sum(s["class_counts"].values()) == 6 * 25
        assert 0.0 <= s["full_coverage_share"] <= 1.0
        # при малых выборках хи-квадрат конечный и неотрицательный
        assert s["vertex_chi2_df"] >= 0.0
        assert s["quadrant_chi2_df"] >= 0.0

    def test_chi2_uniform(self):
        assert chi2_uniform([10, 10, 10, 10], 10.0) == 0.0
        assert chi2_uniform([0, 40], 20.0) == 40.0
        assert chi2_uniform([1, 2, 3], 0.0) == 0.0

    def test_single_workers_path(self):
        result = run_batch(make_config(objects__count=3), n_runs=2, workers=1, seed_start=50)
        assert [r["seed"] for r in result["runs"]] == [50, 51]
        assert all(r["ok"] for r in result["runs"])


class TestCsvAndCharts:
    def test_write_csv(self, batch, tmp_path):
        paths = write_batch_csv(batch, tmp_path)
        assert set(paths) == {"runs_summary.csv", "objects_all.csv"}

        with paths["runs_summary.csv"].open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert all(r["ok"] == "True" for r in rows)

        with paths["objects_all.csv"].open(encoding="utf-8") as f:
            obj_rows = list(csv.DictReader(f))
        assert len(obj_rows) == 6 * 25
        assert {r["seed"] for r in obj_rows} == {"100", "101", "102", "103", "104", "105"}

    def test_render_charts(self, batch, tmp_path):
        paths = render_report_charts(batch, tmp_path)
        assert set(paths) == {
            "chart_placement.png", "chart_attributes.png",
            "chart_geometry.png", "chart_frames_space.png",
        }
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0
