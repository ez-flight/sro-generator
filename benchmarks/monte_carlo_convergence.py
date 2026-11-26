"""Исследование Монте-Карло: сходимость средней геометрической полноты покрытия района.

Каждый из M независимых запусков генерирует новую случайную конфигурацию
объектов СРО (seed подряд от начального), моделирует оптическую недоступность
части объектов (облачность) и досъёмку РСА с фиксированным вычислительным
бюджетом алгоритма (K итераций выбора кадров). Для накопленной выборки
по контрольным точкам m считаются среднее, std, SEM и 95%-й ДИ двух метрик:
полноты досъёмки объектов, недоступных оптике, и итоговой геометрической
полноты района (оптика + РСА).

Основной сценарий тот же, что в convergence_analysis.py (район 25×25 км,
60 объектов, 240 кадров-кандидатов 5×5 км, 4 азимута пролёта), чтобы
результаты сопоставимы с исследованием сходимости алгоритма (версия 1.2.0).

Запуск:
    .venv/bin/python benchmarks/monte_carlo_convergence.py                # M=1000, K=70
    .venv/bin/python benchmarks/monte_carlo_convergence.py --quick       # дымовой прогон

Результаты: CSV/JSON/PNG в --output (по умолчанию output/monte_carlo).
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
    METRIC_GEOMETRIC,
    METRIC_HIDDEN,
    MonteCarloParams,
    render_monte_carlo_charts,
    run_monte_carlo_study,
    write_monte_carlo_csv,
    write_monte_carlo_json,
)


def research_config() -> dict:
    """Фиксированные параметры сценария (как в convergence_analysis.py, seed задаётся запуском)."""
    return {
        "name": "monte_carlo_research",
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
        description="Метод Монте-Карло: сходимость средней геометрической полноты покрытия района")
    parser.add_argument("--experiments", type=int, default=1000, help="число независимых запусков M")
    parser.add_argument("--iterations", type=int, default=70, help="бюджет алгоритма досъёмки K (итераций)")
    parser.add_argument("--seed-start", type=int, default=0, help="начальный seed сценариев")
    parser.add_argument("--cloud-fraction", type=float, default=0.4,
                        help="вероятность оптической недоступности объекта (облачность)")
    parser.add_argument("--selection", choices=["greedy", "softmax", "random"], default="greedy",
                        help="правило выбора кадра досъёмки")
    parser.add_argument("--temperature", type=float, default=0.25, help="температура softmax")
    parser.add_argument("--precision", type=float, default=0.01,
                        help="требуемая точность оценки средней полноты (доля)")
    parser.add_argument("--workers", type=int, default=4, help="параллельных процессов")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "monte_carlo",
                        help="каталог для CSV/JSON/PNG")
    parser.add_argument("--quick", action="store_true", help="дымовой прогон: M=40, K=30")
    args = parser.parse_args(argv)
    if args.quick:
        args.experiments = min(args.experiments, 40)
        args.iterations = min(args.iterations, 30)

    checkpoints = tuple(m for m in (10, 25, 50, 100, 200, 500, 1000) if m <= args.experiments)
    if checkpoints[-1] != args.experiments:
        checkpoints = checkpoints + (args.experiments,)
    params = MonteCarloParams(
        experiments=args.experiments,
        iterations=args.iterations,
        checkpoints=checkpoints,
        seed_start=args.seed_start,
        cloud_fraction=args.cloud_fraction,
        selection=args.selection,
        temperature=args.temperature,
        precision=args.precision,
    )

    started = time.perf_counter()
    study = run_monte_carlo_study(
        research_config(), params, workers=args.workers,
        progress=lambda done, total: (
            done % 50 == 0 or done == total) and print(f"  запусков: {done}/{total}", flush=True),
    )
    elapsed = time.perf_counter() - started

    paths = write_monte_carlo_csv(study, args.output)
    paths["monte_carlo.json"] = write_monte_carlo_json(study, args.output)
    try:
        paths.update(render_monte_carlo_charts(study, args.output))
    except Exception as exc:  # графики — вторичны: исследование уже сохранено в CSV/JSON
        print(f"предупреждение: построение графиков завершилось ошибкой: {exc}")

    print(f"\nИсследование завершено за {elapsed:.1f} c. Итоги (M = {args.experiments}, K = {args.iterations}, "
          f"облачность {args.cloud_fraction:.0%}, досъёмка {args.selection}):")
    for metric, title in ((METRIC_GEOMETRIC, "итоговая геометрическая полнота G"),
                          (METRIC_HIDDEN, "полнота досъёмки скрытых объектов C")):
        last = study["running_stats"][metric][-1]
        req = study["required_experiments"][metric]
        a = study["assessment"][metric]
        print(f"  {title}: среднее {last['mean']:.4f} ± {last['ci95_half']:.4f} (95%-й ДИ), "
              f"std {last['std']:.4f}; n* для ±{args.precision:.3g} = {req['n_required']}; "
              f"среднее сошлось: {a['mean_converged']}, неопределённость убывает: {a['uncertainty_decreases']}")
    for w in study["warnings"]:
        print(f"  предупреждение: {w}")
    print("Файлы:")
    for name, p in sorted(paths.items()):
        print(f"  {name} -> {p}")

    # краткая сводка для встраивания в отчёт
    summary = {
        "elapsed_s": round(elapsed, 1),
        "params": study["params"],
        "final": {m: study["running_stats"][m][-1] for m in (METRIC_GEOMETRIC, METRIC_HIDDEN)},
        "required": study["required_experiments"],
        "assessment": study["assessment"],
    }
    (args.output / "monte_carlo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
