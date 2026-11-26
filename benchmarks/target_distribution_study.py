"""Статистическое исследование законов пространственного размещения целевых объектов.

Для каждого из четырёх законов (равномерно-случайное, групповое, регулярное,
линейное) выполняется N независимых статистических испытаний — реализаций
модели объектов съёмки на одном и том же районе 25×25 км (60 объектов,
параметры формы как в исследовательском сценарии). По каждой реализации
вычисляются пространственные характеристики (среднее расстояние до ближайшего
соседа с поправкой на краевые эффекты, индекс Кларка–Эванса R, индекс
дисперсии квадрантного счёта I, осевой индекс анизотропии направлений,
доля покрытия района); по накопленным выборкам строятся классические оценки
M*(n), D̃(n), σ̃(n), ε(β,n) и доверительные интервалы при β = 0.95.

Запуск:
    .venv/bin/python benchmarks/target_distribution_study.py            # N=200 на закон
    .venv/bin/python benchmarks/target_distribution_study.py --quick   # дымовой прогон

Результаты: CSV/JSON/PNG в --output (по умолчанию output/target_distribution).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sro_generator.geometry import LocalCRS, build_district_polygon  # noqa: E402
from sro_generator.target_model import (  # noqa: E402
    MODELS,
    MODEL_TITLES_RU,
    SpatialModelParams,
    generate_target_objects,
    render_target_charts,
    run_target_model_study,
    write_target_csv,
    write_target_json,
)


def research_district():
    """Исследовательский район 25×25 км (как в convergence/monte-carlo стендах)."""
    local = LocalCRS(55.0, 60.0)
    return build_district_polygon(local, 25_000.0, 25_000.0, 0.0), local


def research_params() -> SpatialModelParams:
    """Параметры модели объектов съёмки.

    Объекты 0,05–0,5 км² (характерный масштаб целевых объектов ДЗЗ — здания,
    сооружения, узлы инфраструктуры), 60 объектов на районе 25×25 км; при
    больших площадях ограничение непересечения регуляризует кластерное и
    линейное размещение и подавляет сигнатуры законов.
    """
    return SpatialModelParams(
        count=60,
        vertices_min=3,
        vertices_max=10,
        area_min_m2=50_000.0,
        area_max_m2=500_000.0,
        irregularity=0.35,
        max_aspect_ratio=2.5,
        boundary_margin_m=500.0,
        min_distance_m=0.0,
        n_clusters=5,
        cluster_sigma_m=1_500.0,
        jitter_fraction=0.6,
        n_lines=4,
        line_sigma_m=400.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="дымовой прогон (N=20)")
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "output" / "target_distribution"))
    args = parser.parse_args()

    realizations = args.realizations or (20 if args.quick else 200)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    district, _local = research_district()
    params = research_params()

    print(f"Район 25×25 км (площадь {district.area / 1e6:.0f} км²), объектов {params.count}, "
          f"реализаций на закон: {realizations}, seeds {args.seed_start}…"
          f"{args.seed_start + realizations - 1}")
    t0 = time.time()
    study = run_target_model_study(
        params, district,
        realizations=realizations,
        seed_start=args.seed_start,
        beta=0.95,
        grid=5,
        models=MODELS,
        progress=lambda done, total: print(f"\r  испытаний: {done}/{total}", end="", flush=True),
    )
    print(f"\nИсследование завершено за {time.time() - t0:.1f} с")

    # CSV/JSON пишутся до графиков, чтобы сбой отрисовки не терял результаты
    csv_paths = write_target_csv(study, out)
    json_path = write_target_json(study, out)
    print("CSV:", ", ".join(str(p) for p in csv_paths.values()))
    print("JSON:", json_path)

    examples = {}
    for model in MODELS:
        p = SpatialModelParams(**{**study["per_model"][model]["params"]})
        rng = np.random.default_rng(np.random.SeedSequence([args.seed_start, 20261126, len(model)]))
        examples[model] = generate_target_objects(p, district, rng)
    chart_paths = render_target_charts(study, examples, out)
    print("PNG:", ", ".join(str(p) for p in chart_paths.values()))

    # сводка в консоль
    print("\nИтоговые оценки (β = 0.95):")
    header = f"{'закон':28s} {'метрика':22s} {'M*(n)':>10s} {'σ̃(n)':>10s} {'ε(β,n)':>10s}"
    print(header)
    for model in MODELS:
        pm = study["per_model"][model]
        for metric in study["metrics"]:
            last = pm["classical"][metric][-1]
            print(f"{MODEL_TITLES_RU[model]:28s} {metric:22s} "
                  f"{last['m_star']:10.4f} {last['sigma_tilde']:10.4f} {last['epsilon']:10.4f}")
        print(f"{'':28s} {'доля отклонений CSR: z-тест':22s} {pm['ce_z_reject_share']:10.2f}")
        print(f"{'':28s} {'доля отклонений CSR: χ²':22s} {pm['csr_chi2_reject_share']:10.2f}")

    summary = {
        "realizations": realizations,
        "seed_start": args.seed_start,
        "duration_note": "см. вывод стенда",
        "conclusion_shares": {
            m: {
                "ce_z_reject_share": study["per_model"][m]["ce_z_reject_share"],
                "csr_chi2_reject_share": study["per_model"][m]["csr_chi2_reject_share"],
            } for m in MODELS
        },
    }
    (out / "target_study_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
