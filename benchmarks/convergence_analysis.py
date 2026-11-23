"""Исследование сходимости алгоритма планирования кадров РСА (версия 1.2.0).

Серия экспериментов на одном фиксированном сценарии (район 25×25 км,
60 объектов, 240 кадров-кандидатов 5×5 км, 4 азимута, seed=42):

- основное исследование: M независимых экспериментов × K итераций
  (softmax, T=0.25, seeds подряд от начального);
- температурная сетка T ∈ {0.05, 0.25, 1.0, 4.0} — влияние температуры
  на скорость сходимости и устойчивость;
- сравнение правил выбора кадра: детерминированный жадный (greedy),
  рандомизированный жадный (softmax) и равномерно случайный (random).

Статистический анализ по фактическим траекториям I(k):
- итерации достижения порогов 90/95/99 % покрытия с вероятностью ≥ 95 %;
- тренд межэкспериментального разброса std(k): ρ Спирмена + перестановочный
  тест (сходимость не предполагается заранее).

Запуск:
    .venv/bin/python benchmarks/convergence_analysis.py              # M=100, K=200
    .venv/bin/python benchmarks/convergence_analysis.py --quick      # дымовой прогон

Результаты: CSV/JSON/PNG в --output (по умолчанию output/convergence).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sro_generator.convergence import (  # noqa: E402
    ConvergenceParams,
    coverage_threshold_analysis,
    render_comparison_charts,
    render_convergence_charts,
    run_convergence_study,
    std_trend_analysis,
    write_convergence_csv,
    write_convergence_json,
)

TEMPERATURES = (0.05, 0.25, 1.0, 4.0)
THRESHOLDS = (0.90, 0.95, 0.99)


def research_config() -> dict:
    """Фиксированный сценарий исследования: все 240 кадров покрывают район на 99.7 %."""
    return {
        "name": "convergence_research",
        "seed": 42,
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


def _auc_mean_curve(study: dict) -> float:
    """Нормированная площад под средней кривой I(k) — интегральная скорость роста."""
    import numpy as np

    mean = np.asarray([s["mean"] for s in study["iteration_stats"]], dtype=float)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(mean, dx=1.0) / len(mean))


def analyze_study(study: dict, n_perm: int) -> dict:
    """Сводный анализ одного исследования: диагноз, пороги, тренд разброса."""
    thresholds = coverage_threshold_analysis(study, thresholds=THRESHOLDS, confidence=0.95)
    trend = std_trend_analysis(study["iteration_stats"], n_perm=n_perm)
    final = study["iteration_stats"][-1]
    return {
        "verdict": study["convergence"]["verdict"],
        "verdict_ru": study["convergence"]["verdict_ru"],
        "stabilized_at_iteration": study["convergence"]["stabilized_at_iteration"],
        "final_mean": final["mean"],
        "final_std": final["std"],
        "final_ci95_half": (final["ci95_high"] - final["ci95_low"]) / 2.0,
        "final_band_p10_p90": study["convergence"]["final_band_p10_p90"],
        "auc_mean_curve": _auc_mean_curve(study),
        "thresholds": thresholds,
        "std_trend": trend,
        "warnings": study.get("warnings", []),
    }


def run_research(args: argparse.Namespace) -> dict:
    """Выполнить все исследования и вернуть пакет результатов."""
    config = research_config()
    bundle: dict = {
        "config": config,
        "experiments": args.experiments,
        "iterations": args.iterations,
        "seed_start": args.seed_start,
        "studies": {},
        "analyses": {},
    }

    def _run(key: str, label: str, **overrides) -> dict:
        params = ConvergenceParams(
            experiments=args.experiments,
            iterations=args.iterations,
            seed_start=args.seed_start,
            **overrides,
        )
        started = time.perf_counter()
        study = run_convergence_study(config, params, workers=args.workers)
        study["label"] = label
        bundle["studies"][key] = study
        bundle["analyses"][key] = analyze_study(study, n_perm=args.n_perm)
        print(f"[{time.perf_counter() - started:7.1f} c] {label}: "
              f"final_mean={study['iteration_stats'][-1]['mean']:.4f}, "
              f"verdict={study['convergence']['verdict']}", flush=True)
        return study

    # 1. Основное исследование (softmax T=0.25)
    _run("softmax_t0.25", "softmax T=0.25 (основное)", selection="softmax", temperature=0.25)

    # 2. Температурная сетка (T=0.25 уже есть)
    for t in TEMPERATURES:
        if t == 0.25:
            continue
        key = f"softmax_t{t}"
        _run(key, f"softmax T={t}", selection="softmax", temperature=t)

    # 3. Сравнение правил выбора (greedy детерминирован — достаточно 2 прогонов)
    greedy_experiments = 2 if args.experiments >= 2 else args.experiments
    saved = args.experiments
    args.experiments = greedy_experiments
    _run("greedy", "greedy (детерминированный)", selection="greedy")
    args.experiments = saved
    _run("random", "random (равномерный)", selection="random")

    return bundle


def write_artifacts(bundle: dict, out_dir: Path) -> dict[str, Path]:
    """Сохранить CSV/JSON/PNG: основное исследование целиком + сравнение."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    main = bundle["studies"]["softmax_t0.25"]
    for name, p in write_convergence_csv(main, out_dir).items():
        paths[name] = p
    paths["convergence.json"] = write_convergence_json(main, out_dir)
    for name, p in render_convergence_charts(main, out_dir).items():
        paths[name] = p

    # Сравнительные графики: кривые среднего/std + сводные метрики
    def _label(key: str) -> str:
        sel = bundle["studies"][key]["params"]["selection"]
        temp = bundle["studies"][key]["params"]["temperature"]
        return f"{sel} T={temp}" if sel == "softmax" else sel

    cmp_order = ["greedy", "softmax_t0.25", "random",
                 "softmax_t0.05", "softmax_t1.0", "softmax_t4.0"]
    cmp_order = [k for k in cmp_order if k in bundle["studies"]]
    for name, p in render_comparison_charts(
        [(_label(k), bundle["studies"][k]) for k in cmp_order], out_dir
    ).items():
        paths[name] = p

    # CSV сводки по всем исследованиям
    summary_path = out_dir / "convergence_comparison_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "study", "selection", "temperature", "verdict", "stabilized_at_iteration",
            "final_mean", "final_ci95_half", "final_std", "final_band_p10_p90", "auc_mean_curve",
            "k_conf_90", "k_conf_95", "k_conf_99",
            "share_90_final", "share_95_final", "share_99_final",
            "std_trend_rho", "std_trend_p_value", "std_decreases",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in cmp_order:
            study = bundle["studies"][key]
            a = bundle["analyses"][key]
            th = {r["threshold"]: r for r in a["thresholds"]}
            writer.writerow({
                "study": key,
                "selection": study["params"]["selection"],
                "temperature": study["params"]["temperature"],
                "verdict": a["verdict"],
                "stabilized_at_iteration": a["stabilized_at_iteration"],
                "final_mean": round(a["final_mean"], 8),
                "final_ci95_half": round(a["final_ci95_half"], 8),
                "final_std": round(a["final_std"], 8),
                "final_band_p10_p90": round(a["final_band_p10_p90"], 8),
                "auc_mean_curve": round(a["auc_mean_curve"], 8),
                "k_conf_90": th[0.90]["k_confidence"],
                "k_conf_95": th[0.95]["k_confidence"],
                "k_conf_99": th[0.99]["k_confidence"],
                "share_90_final": round(th[0.90]["reach_share_final"], 6),
                "share_95_final": round(th[0.95]["reach_share_final"], 6),
                "share_99_final": round(th[0.99]["reach_share_final"], 6),
                "std_trend_rho": round(a["std_trend"]["spearman_rho_std"], 6),
                "std_trend_p_value": round(a["std_trend"]["permutation_p_value"], 6),
                "std_decreases": a["std_trend"]["decreasing_significant"],
            })
    paths["convergence_comparison_summary.csv"] = summary_path

    # JSON: конфигурация + анализы всех исследований (траектории — в convergence.json основного)
    analysis_path = out_dir / "convergence_analysis.json"
    analysis_path.write_text(json.dumps({
        "config": bundle["config"],
        "experiments": bundle["experiments"],
        "iterations": bundle["iterations"],
        "seed_start": bundle["seed_start"],
        "analyses": bundle["analyses"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["convergence_analysis.json"] = analysis_path

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Исследование сходимости алгоритма планирования кадров РСА")
    parser.add_argument("--experiments", type=int, default=100, help="число независимых экспериментов M")
    parser.add_argument("--iterations", type=int, default=200, help="число итераций K в эксперименте")
    parser.add_argument("--seed-start", type=int, default=0, help="начальный seed экспериментов")
    parser.add_argument("--workers", type=int, default=4, help="параллельных процессов")
    parser.add_argument("--n-perm", type=int, default=10_000, help="перестановок в тесте тренда std")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "convergence",
                        help="каталог для CSV/JSON/PNG")
    parser.add_argument("--quick", action="store_true", help="дымовой прогон: M=12, K=60")
    args = parser.parse_args(argv)
    if args.quick:
        args.experiments = min(args.experiments, 12)
        args.iterations = min(args.iterations, 60)
        args.n_perm = min(args.n_perm, 1000)

    started = time.perf_counter()
    bundle = run_research(args)
    paths = write_artifacts(bundle, args.output)
    print(f"\nИсследование завершено за {time.perf_counter() - started:.1f} c. Файлы:")
    for name, p in sorted(paths.items()):
        print(f"  {name} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
