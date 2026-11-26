"""Влияние облачности и бюджета итераций K на полноту покрытия района (1.3.0).

Сетка «облачность × K × правило выбора»: облачность ∈ {10, 20, 40, 60, 80 %},
K ∈ {5, 10, 20, 30, 50, 70}, правила greedy и softmax (T=0.25). В каждой ячейке
M ≥ 200 независимых запусков Монте-Карло с одинаковым диапазоном seed
(спаренные сценарии между ячейками). Для каждой ячейки: среднее/std/SEM/95 %-й
ДИ метрик G и C, вероятности достижения порогов 90/95/99 %, минимальное K
достижения порогов с вероятностью ≥ 95 %. Проверяется математическая
согласованность модели облачности (verify_cloud_model).

Запуск:
    .venv/bin/python benchmarks/cloud_coverage_analysis.py                 # полная сетка, M=200
    .venv/bin/python benchmarks/cloud_coverage_analysis.py --quick        # дымовой прогон

Результаты: CSV/JSON/PNG в --output (по умолчанию output/cloud_coverage).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sro_generator.monte_carlo import (  # noqa: E402
    compute_combo_stats,
    min_k_for_thresholds,
    render_sensitivity_charts,
    run_cloud_sensitivity,
    verify_cloud_model,
    write_sensitivity_csv,
    write_sensitivity_json,
)

CLOUDS = (0.10, 0.20, 0.40, 0.60, 0.80)
K_VALUES = (5, 10, 20, 30, 50, 70)
SELECTIONS = ("greedy", "softmax")
THRESHOLDS = (0.90, 0.95, 0.99)


def research_config() -> dict:
    """Тот же класс сценариев, что в предыдущих исследованиях (25×25 км, 60 объектов, 240 кадров)."""
    return {
        "name": "cloud_coverage_research",
        "district": {"center_lat": 55.0, "center_lon": 60.0,
                     "width_km": 25.0, "height_km": 25.0, "rotation_deg": 0.0},
        "objects": {
            "count": 60,
            "vertices_min": 3,
            "vertices_max": 10,
            "area_min_m2": 500_000.0,
            "area_max_m2": 5_000_000.0,
            "irregularity": 0.35,
            "rotation_min_deg": 0.0,
            "rotation_max_deg": 360.0,
            "max_aspect_ratio": 2.5,
        },
        "placement": {
            "mode": "random",
            "min_distance_m": 500.0,
            "boundary_margin_m": 100.0,
            "allow_overlap": False,
            "max_overlap_fraction": 0.0,
            "max_attempts_per_object": 10_000,
            "manual_centers": [],
        },
        "attributes": {
            "class_labels": ["warehouse", "industrial", "transport"],
            "priority_min": 1,
            "priority_max": 5,
            "deadline_choices_h": [30, 48],
            "resolution_choices_m": [0.5, 1.0],
        },
        "frames": {
            "enabled": True,
            "along_track_km": 5.0,
            "cross_track_km": 5.0,
            "pass_azimuths_deg": [30.0, 120.0, 210.0, 300.0],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Влияние облачности и K на полноту покрытия района (метод Монте-Карло)")
    parser.add_argument("--experiments", type=int, default=200, help="запусков M в каждой ячейке сетки")
    parser.add_argument("--seed-start", type=int, default=0, help="начальный seed (общий для всех ячеек)")
    parser.add_argument("--workers", type=int, default=4, help="параллельных процессов")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "cloud_coverage",
                        help="каталог для CSV/JSON/PNG")
    parser.add_argument("--quick", action="store_true",
                        help="дымовой прогон: облачность {20,60 %}, K {10,30,70}, M=20")
    args = parser.parse_args(argv)
    clouds, k_values = CLOUDS, K_VALUES
    if args.quick:
        clouds = (0.20, 0.60)
        k_values = (10, 30, 70)
        args.experiments = min(args.experiments, 20)

    started = time.perf_counter()
    study = run_cloud_sensitivity(
        research_config(),
        cloud_fractions=clouds,
        k_values=k_values,
        selections=SELECTIONS,
        experiments=args.experiments,
        seed_start=args.seed_start,
        workers=args.workers,
        progress=lambda done, total: (
            done % 200 == 0 or done == total) and print(f"  запусков: {done}/{total}", flush=True),
    )
    for cell in study["cells"]:
        cell["stats"] = compute_combo_stats(cell["experiments"], thresholds=THRESHOLDS)
    checks = verify_cloud_model(study)
    min_k = min_k_for_thresholds(study, thresholds=THRESHOLDS, confidence=0.95)
    study["verification"] = checks
    study["min_k"] = min_k
    elapsed = time.perf_counter() - started

    paths = write_sensitivity_csv(study, args.output)
    paths["cloud_coverage.json"] = write_sensitivity_json(study, args.output)
    try:
        paths.update(render_sensitivity_charts(study, args.output))
    except Exception as exc:  # графики вторичны: исследование уже сохранено
        print(f"предупреждение: построение графиков завершилось ошибкой: {exc}")

    print(f"\nСетка завершена за {elapsed:.1f} c. Проверки модели:")
    for name, c in checks.items():
        print(f"  {name}: ok={c['ok']}", flush=True)
    print("Итоги (G: среднее ± 95 %-й ДИ):")
    for cell in study["cells"]:
        s = cell["stats"]["g"]
        print(f"  обл={cell['cloud_fraction']:.0%} K={cell['k']:>2} {cell['selection']:<7} "
              f"G={s['mean']:.4f}±{s['ci95_half']:.4f} C={cell['stats']['c']['mean']:.4f}", flush=True)
    print("Минимальное K для порогов G (вероятность ≥ 95 %):")
    for row in min_k:
        print(f"  обл={row['cloud_fraction']:.0%} {row['selection']:<7} "
              f"γ={row['threshold']:.2f}: K={row['k_min']}", flush=True)
    print("Файлы:")
    for name, p in sorted(paths.items()):
        print(f"  {name} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
