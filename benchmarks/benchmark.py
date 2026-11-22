"""Бенчмарк версии 1.1.0: старый и новый алгоритмы размещения на одинаковых конфигурациях.

Сравниваются четыре комбинации:
- legacy+brute  — поведение 1.0.0 (область центров через описанный радиус, полный перебор пар);
- legacy+strtree — область 1.0.0 + пространственный индекс;
- exact+brute   — точная область центров + полный перебор;
- exact+strtree — точная область + STRtree (конфигурация 1.1.0 по умолчанию).

Также проверяется эквивалентность решений STRtree и полного перебора
(идентичные отпечатки результата при одинаковом seed).

Запуск:
    .venv/bin/python benchmarks/benchmark.py                 # 100/500/1000/5000
    .venv/bin/python benchmarks/benchmark.py --counts 100 500 --output benchmarks/results

Результаты: CSV в --output и графики (render_benchmark_charts).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sro_generator.config import ScenarioConfig  # noqa: E402
from sro_generator.generator import generate_scenario  # noqa: E402
from sro_generator.stats import render_benchmark_charts  # noqa: E402

COMBOS = [
    ("legacy+brute", "legacy", False),
    ("legacy+strtree", "legacy", True),
    ("exact+brute", "exact", False),
    ("exact+strtree", "exact", True),
]


def build_config(count: int, seed: int) -> dict:
    """Конфигурация бенчмарка: район 300×300 км, мелкие объекты, без кадров."""
    return {
        "name": f"benchmark_{count}",
        "seed": seed,
        "district": {"center_lat": 55.0, "center_lon": 60.0,
                     "width_km": 300.0, "height_km": 300.0, "rotation_deg": 0.0},
        "objects": {
            "count": count,
            "vertices_min": 3,
            "vertices_max": 10,
            "area_min_m2": 50_000.0,
            "area_max_m2": 200_000.0,
            "irregularity": 0.35,
            "rotation_min_deg": 0.0,
            "rotation_max_deg": 360.0,
            "max_aspect_ratio": 2.5,
        },
        "placement": {
            "mode": "random",
            "min_distance_m": 100.0,
            "boundary_margin_m": 50.0,
            "allow_overlap": False,
            "max_overlap_fraction": 0.0,
            "max_attempts_per_object": 100_000,
            "manual_centers": [],
            "center_region_mode": "exact",
            "spatial_index": True,
        },
        "attributes": {
            "class_labels": ["warehouse", "industrial", "transport"],
            "priority_min": 1,
            "priority_max": 5,
            "deadline_choices_h": [30.0, 48.0],
            "resolution_choices_m": [0.5, 1.0],
        },
        "frames": {"enabled": False},
    }


def fingerprint(scenario) -> tuple:
    return (
        len(scenario.objects),
        round(sum(o.area_m2 for o in scenario.objects), 6),
        round(sum(o.center.x for o in scenario.objects), 3),
        round(sum(o.center.y for o in scenario.objects), 3),
        scenario.report.attempts_total,
    )


def run_one(count: int, seed: int, combo: str, region_mode: str, index: bool) -> dict:
    cfg_dict = build_config(count, seed)
    cfg_dict["placement"]["center_region_mode"] = region_mode
    cfg_dict["placement"]["spatial_index"] = index
    cfg = ScenarioConfig.model_validate(cfg_dict)
    row: dict = {"objects_count": count, "seed": seed, "combo": combo}
    tracemalloc.start()
    started = time.perf_counter()
    try:
        scenario = generate_scenario(cfg)
        row["ok"] = True
        row["error"] = ""
        row["n_objects"] = len(scenario.objects)
        row["attempts_total"] = scenario.report.attempts_total
        row["exact_checks_total"] = scenario.report.exact_checks_total
        row["tree_queries"] = scenario.report.tree_queries
        row["tree_rebuilds"] = scenario.report.tree_rebuilds
        row["fill_ratio"] = round(
            sum(o.area_m2 for o in scenario.objects) / scenario.district_local.area, 8
        )
        row["fingerprint"] = fingerprint(scenario)
        row["region_area_mean_m2"] = round(scenario.report.center_region_area_mean_m2, 1)
    except Exception as exc:  # noqa: BLE001 — бенчмарк фиксирует отказ как данные
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["n_objects"] = 0
        row["attempts_total"] = ""
        row["exact_checks_total"] = ""
        row["tree_queries"] = ""
        row["tree_rebuilds"] = ""
        row["fill_ratio"] = ""
        row["fingerprint"] = None
        row["region_area_mean_m2"] = ""
    finally:
        row["duration_s"] = round(time.perf_counter() - started, 4)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row["peak_mem_mb"] = round(peak / (1024 * 1024), 2)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк СРО 1.1.0")
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 500, 1000, 5000])
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for count in args.counts:
        # одинаковый seed для всех комбинаций при данном count — честное сравнение
        seed = args.seed + count
        for combo, region_mode, index in COMBOS:
            row = run_one(count, seed, combo, region_mode, index)
            rows.append(row)
            status = "ok" if row["ok"] else f"FAIL: {row['error']}"
            print(f"[{combo:16s}] n={count:5d}  {row['duration_s']:8.3f} с  "
                  f"checks={row['exact_checks_total']!s:>10}  mem={row['peak_mem_mb']:7.2f} МБ  {status}",
                  flush=True)

    # эквивалентность STRtree и полного перебора при одинаковом seed
    by_key: dict[tuple[int, str], tuple] = {}
    for row in rows:
        if row["ok"]:
            by_key[(row["objects_count"], row["combo"])] = row["fingerprint"]
    for count in args.counts:
        for region_mode in ("legacy", "exact"):
            fp_tree = by_key.get((count, f"{region_mode}+strtree"))
            fp_brute = by_key.get((count, f"{region_mode}+brute"))
            eq = (fp_tree is not None and fp_tree == fp_brute)
            for row in rows:
                if row["objects_count"] == count and row["combo"].startswith(region_mode):
                    row["equivalence_ok"] = eq
            print(f"эквивалентность {region_mode}: n={count} -> {'СОВПАЛО' if eq else 'РАСХОЖДЕНИЕ'}",
                  flush=True)

    csv_path = args.output / "benchmark_1_1.csv"
    fields = ["objects_count", "seed", "combo", "ok", "error", "duration_s", "peak_mem_mb",
              "n_objects", "attempts_total", "exact_checks_total", "tree_queries",
              "tree_rebuilds", "fill_ratio", "region_area_mean_m2", "equivalence_ok"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"CSV: {csv_path}")

    chart_paths = render_benchmark_charts(rows, args.output)
    for name, p in chart_paths.items():
        print(f"график: {name} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
