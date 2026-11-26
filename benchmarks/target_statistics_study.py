"""Статистическое исследование объектов съёмки: проверка законов размещения
и сходимость статистических оценок их характеристик (вкладка 4).

Использует вычислительное ядро `sro_generator/target_model.py` и параметры
исследовательского сценария из `benchmarks/target_distribution_study.py`
(район 25×25 км, 60 объектов 0,05–0,5 км²). Одно статистическое испытание —
одна независимая реализация пространственного распределения объектов.

Выполняет:
1. основное исследование: N = 200 независимых реализаций каждого из четырёх
   законов (seeds 0…199 — те же, что в target_distribution_study.py, поэтому
   реализации совпадают с сохранёнными 800 реализациями 2.1.0); классические
   оценки M*(n), D̃(n), σ̃(n), ε(β,n) при β = 0.95; пул расстояний до
   ближайшего соседа для F*(r);
2. независимый имитационный эталон F(r) — те же законы, seeds 100000…100199;
3. эксперимент о влиянии геометрических ограничений на соответствие CSR
   (расхождение R = 1,025 ± 0,009, не накрывающее 1): равномерно-случайный
   закон с масштабом площадей объектов ×1, ×0.1, ×0.01 и почти точечными
   объектами (1–10 м²) — при стремлении объектов к точкам запрет пересечений
   исчезает и R → 1;
4. шесть обязательных графиков (PNG + SVG) и две таблицы (CSV).

Запуск:
    .venv/bin/python benchmarks/target_statistics_study.py            # N=200
    .venv/bin/python benchmarks/target_statistics_study.py --quick   # дымовой N=20

Результаты: output/target_statistics/.
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

from sro_generator.geometry import build_district_polygon, LocalCRS  # noqa: E402
from sro_generator.target_model import (  # noqa: E402
    MODELS,
    MODEL_TITLES_RU,
    SpatialModelParams,
    collect_nn_samples,
    generate_target_objects,
    render_statistics_charts,
    run_target_model_study,
    sample_skewness,
    write_target_csv,
    write_target_json,
)
from benchmarks.target_distribution_study import research_district, research_params  # noqa: E402

SELECTED_MODEL = "uniform"
SELECTED_METRIC = "nn_mean_m"
EPSILON0 = 5.0  # м, заданная точность для графика 6
BETA = 0.95


def csr_discrepancy_experiment(district, realizations: int, seed_start: int = 0) -> list[dict]:
    """Влияние запрета пересечений на соответствие равномерного закона CSR.

    Масштабируем площади объектов: при уменьшении объектов до почти точечных
    ограничение непересечения перестаёт влиять на центры, и индекс
    Кларка–Эванса должен стремиться к теоретической единице CSR.
    """
    base = research_params()
    rows = []
    for scale, label in ((1.0, "0,05–0,5 км² (базовый)"),
                         (0.1, "0,005–0,05 км²"),
                         (0.01, "500–5000 м²"),
                         (None, "1–10 м² (почти точечные)")):
        if scale is None:
            p = SpatialModelParams(**{**base.__dict__, "model": "uniform",
                                      "area_min_m2": 1.0, "area_max_m2": 10.0})
        else:
            p = SpatialModelParams(**{**base.__dict__, "model": "uniform",
                                      "area_min_m2": base.area_min_m2 * scale,
                                      "area_max_m2": base.area_max_m2 * scale})
        study = run_target_model_study(p, district, realizations=realizations,
                                       seed_start=seed_start, beta=BETA,
                                       models=("uniform",))
        last = study["per_model"]["uniform"]["classical"]["ce_r"][-1]
        rows.append({
            "areas": label,
            "area_min_m2": p.area_min_m2,
            "area_max_m2": p.area_max_m2,
            "m_star": last["m_star"],
            "epsilon": last["epsilon"],
            "ci_low": last["ci_low"],
            "ci_high": last["ci_high"],
            "ci_contains_1": bool(last["ci_low"] <= 1.0 <= last["ci_high"]),
        })
        print(f"  площади {label:28s} R = {last['m_star']:.4f} ± {last['epsilon']:.4f} "
              f"({'накрывает 1' if rows[-1]['ci_contains_1'] else 'НЕ накрывает 1'})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="дымовой прогон (N=20)")
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "output" / "target_statistics"))
    args = parser.parse_args()

    n = args.realizations or (20 if args.quick else 200)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    district, local = research_district()
    params = research_params()
    print(f"Район 25×25 км, объектов {params.count}, реализаций на закон: {n}, "
          f"β = {BETA}, выбрано: закон «{SELECTED_MODEL}», характеристика «{SELECTED_METRIC}», ε₀ = {EPSILON0} м")

    t0 = time.time()
    study = run_target_model_study(
        params, district, realizations=n, seed_start=0, beta=BETA, grid=5,
        progress=lambda d, t: print(f"\r  испытаний: {d}/{t}", end="", flush=True),
    )
    print(f"\nОсновное исследование: {time.time() - t0:.1f} с")

    t0 = time.time()
    nn_empirical = collect_nn_samples(
        params, district, models=MODELS, realizations=n, seed_start=0,
        progress=lambda d, t: print(f"\r  выборки F*(r): {d}/{t}", end="", flush=True),
    )
    nn_reference = collect_nn_samples(
        params, district, models=MODELS, realizations=n, seed_start=100_000,
        progress=lambda d, t: print(f"\r  имитационный эталон: {d}/{t}", end="", flush=True),
    )
    print(f"\nВыборки расстояний: {time.time() - t0:.1f} с "
          f"(эмпирика {[int(v.size) for v in nn_empirical.values()]}, "
          f"эталон {[int(v.size) for v in nn_reference.values()]})")

    print("Эксперимент: влияние запрета пересечений на соответствие CSR")
    csr_rows = csr_discrepancy_experiment(district, realizations=n)

    # данные — до графиков
    csv_paths = write_target_csv(study, out)
    json_path = write_target_json(study, out)
    extra = {
        "selected_model": SELECTED_MODEL,
        "selected_metric": SELECTED_METRIC,
        "epsilon0_m": EPSILON0,
        "beta": BETA,
        "realizations": n,
        "nn_pooled_sizes": {m: int(v.size) for m, v in nn_empirical.items()},
        "nn_reference_sizes": {m: int(v.size) for m, v in nn_reference.items()},
        "skewness": {
            m: {metric: sample_skewness([r[metric] for r in study["per_model"][m]["runs"]])
                for metric in study["metrics"]}
            for m in MODELS
        },
        "csr_discrepancy_experiment": csr_rows,
    }
    (out / "target_statistics_extra.json").write_text(
        json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CSV:", ", ".join(str(p) for p in csv_paths.values()))
    print("JSON:", json_path, "+ target_statistics_extra.json")

    examples = {}
    for model in MODELS:
        p = SpatialModelParams(**study["per_model"][model]["params"])
        rng = np.random.default_rng(np.random.SeedSequence([0, 20261126, len(model)]))
        examples[model] = generate_target_objects(p, district, rng)

    chart_paths = render_statistics_charts(
        study, examples, nn_empirical, nn_reference,
        selected_model=SELECTED_MODEL, metric=SELECTED_METRIC,
        beta=BETA, epsilon0=EPSILON0, out_dir=out, formats=("png", "svg"),
    )
    for name, ps in chart_paths.items():
        print(f"  {name}: {', '.join(p.suffix for p in ps)}")

    print("\nИтоговые оценки (β = 0.95), выбранная характеристика:")
    for model in MODELS:
        last = study["per_model"][model]["classical"][SELECTED_METRIC][-1]
        print(f"  {MODEL_TITLES_RU[model]:26s} M*={last['m_star']:.3f} "
              f"D̃={last['d_tilde']:.4g} σ̃={last['sigma_tilde']:.3f} "
              f"ε={last['epsilon']:.3f} I=[{last['ci_low']:.3f}; {last['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
