"""Статистическое исследование криволинейного распределения объектов.

Стенд версии 2.5.0 для отчёта `CURVED-LINE-STATISTICS.md`. Использует
исследовательский сценарий 25×25 км / 60 объектов 0,05–0,5 км² из
`benchmarks/target_distribution_study.py` и добавляет криволинейное размещение
к четырём базовым законам.

Выполняет:
1. N независимых реализаций каждого из пяти законов (β = 0,95), включая
   прямолинейное/криволинейное сравнение при одинаковых параметрах объектов;
2. пулы F*(r) эмпирические и имитационный эталон (независимые seed);
3. серию извилистостей 0, 1500, 3000, 5000 м на тех же параметрах объектов:
   равномерность вдоль кривой (KS/χ²), поперечное отклонение (KS/χ²),
   влияние на d̄, R, I, R_b и сходимость M*(n), D̃(n), ε(n);
4. существующие шесть научных графиков (PNG + SVG) и таблицы CSV/JSON.

Запуск:
    .venv/bin/python benchmarks/curved_line_statistics.py            # N=200
    .venv/bin/python benchmarks/curved_line_statistics.py --quick   # дымовой N=20

Результаты: output/curved_line/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sro_generator.target_model import (  # noqa: E402
    TARGET_MODELS,
    MODEL_TITLES_RU,
    SpatialModelParams,
    collect_nn_samples,
    generate_target_objects,
    render_statistics_charts,
    run_curved_line_study,
    run_target_model_study,
    write_curved_line_csv,
    write_curved_line_json,
    write_target_csv,
    write_target_json,
)
from benchmarks.target_distribution_study import research_district, research_params  # noqa: E402

BETA = 0.95
CURVE_BEND_M = 2500.0
CURVE_WIDTH_M = 1000.0
BEND_VALUES = (0.0, 1500.0, 3000.0, 5000.0)
SELECTED_MODEL = "curved"
SELECTED_METRIC = "nn_mean_m"
EPSILON0 = 5.0  # м


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="дымовой прогон (N=20)")
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "output" / "curved_line"))
    args = parser.parse_args()

    n = args.realizations or (20 if args.quick else 200)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    district, _local = research_district()
    base = research_params()
    params = SpatialModelParams(**{
        **base.__dict__,
        "curve_bend_m": CURVE_BEND_M,
        "curve_width_m": CURVE_WIDTH_M,
    })
    print(f"Район 25×25 км, объектов {params.count}, N = {n}, β = {BETA}; "
          f"кривая: извилистость {CURVE_BEND_M:g} м, коридор {CURVE_WIDTH_M:g} м; "
          f"сравнение извилистостей: {', '.join(f'{v:g}' for v in BEND_VALUES)} м")

    t0 = time.time()
    study = run_target_model_study(
        params, district, realizations=n, seed_start=0, beta=BETA, grid=5,
        models=TARGET_MODELS,
        progress=lambda d, t: print(f"\r  испытаний: {d}/{t}", end="", flush=True),
    )
    print(f"\nОсновное исследование пяти законов: {time.time() - t0:.1f} с")

    t0 = time.time()
    nn_emp = collect_nn_samples(params, district, models=TARGET_MODELS,
                                realizations=n, seed_start=0,
                                progress=lambda d, t: print(f"\r  F*(r): {d}/{t}", end="", flush=True))
    nn_ref = collect_nn_samples(params, district, models=TARGET_MODELS,
                                realizations=n, seed_start=100_000,
                                progress=lambda d, t: print(f"\r  эталон F(r): {d}/{t}", end="", flush=True))
    print(f"\nПулы расстояний: {time.time() - t0:.1f} с")

    t0 = time.time()
    curve_study = run_curved_line_study(
        params, district, bend_values=BEND_VALUES, realizations=n,
        seed_start=200_000, beta=BETA, bins=10, grid=5,
        progress=lambda d, t: print(f"\r  извилистости: {d}/{t}", end="", flush=True),
    )
    print(f"\nСерия извилистостей ({len(BEND_VALUES)} × {n}): {time.time() - t0:.1f} с")

    # Данные пишем до графиков, чтобы сбой отрисовки не терял результаты прогона.
    target_csv = write_target_csv(study, out)
    target_json = write_target_json(study, out)
    curve_csv = write_curved_line_csv(curve_study, out)
    curve_json = write_curved_line_json(curve_study, out)
    (out / "curved_line_summary.json").write_text(json.dumps({
        "n": n,
        "beta": BETA,
        "curve_bend_m": CURVE_BEND_M,
        "curve_width_m": CURVE_WIDTH_M,
        "bend_values": list(BEND_VALUES),
        "linear_vs_curved": {
            m: {
                metric: study["per_model"][m]["classical"][metric][-1]
                for metric in study["metrics"]
            }
            for m in ("linear", "curved")
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CSV:", ", ".join(p.name for p in list(target_csv.values()) + list(curve_csv.values())))
    print("JSON:", target_json.name, curve_json.name, "+ curved_line_summary.json")

    examples = {}
    for model in TARGET_MODELS:
        p = SpatialModelParams(**study["per_model"][model]["params"])
        rng = np.random.default_rng(np.random.SeedSequence([0, 20261126, len(model)]))
        examples[model] = generate_target_objects(p, district, rng)

    chart_paths = render_statistics_charts(
        study, examples, nn_emp, nn_ref,
        selected_model=SELECTED_MODEL, metric=SELECTED_METRIC,
        beta=BETA, epsilon0=EPSILON0, out_dir=out, formats=("png", "svg"),
    )
    for name, ps in chart_paths.items():
        print(f"  {name}: {', '.join(p.suffix for p in ps)}")

    print("\nПрямолинейное против криволинейного (M* ± ε):")
    for metric in study["metrics"]:
        lin = study["per_model"]["linear"]["classical"][metric][-1]
        cur = study["per_model"]["curved"]["classical"][metric][-1]
        print(f"  {study['metric_titles_ru'][metric]}: "
              f"linear {lin['m_star']:.4f} ± {lin['epsilon']:.4f}; "
              f"curved {cur['m_star']:.4f} ± {cur['epsilon']:.4f}")

    print("\nКриволинейная модель по извилистостям:")
    for item in curve_study["per_bend"]:
        u = item["uniformity"]
        nn = item["classical"]["nn_mean_m"][-1]
        r = item["classical"]["ce_r"][-1]
        print(f"  bend={item['bend_m']:g} м: d̄={nn['m_star']:.1f} ± {nn['epsilon']:.1f} м, "
              f"R={r['m_star']:.3f} ± {r['epsilon']:.3f}, "
              f"KSвд={u['along_ks_mean']:.4f}, |x|={u['lateral_abs_mean_m']:.1f} м")


if __name__ == "__main__":
    main()
