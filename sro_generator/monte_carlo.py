"""Метод Монте-Карло: сходимость оценки средней геометрической полноты покрытия района.

Исследуется статистическая сходимость оценки средней полноты покрытия района при
досъёмке средствами РСА ДЗЗ объектов, недоступных оптическим системам
(например, из-за облачности). В отличие от исследования ``convergence.py``
(фиксированный сценарий, случайность только в алгоритме), здесь случайность —
в самих сценариях: каждый независимый запуск m генерирует новую случайную
конфигурацию объектов СРО из ``seed_start + m``, а параметры моделирования
(район, число объектов, кадры, вычислительный бюджет алгоритма K итераций)
во всех запусках одинаковы.

Модель оптической недоступности: каждый объект независимо от остальных
недоступен оптике с вероятностью ``MonteCarloParams.cloud_fraction``
(облачность над объектом). Оптическая съёмка покрывает все доступные объекты;
объекты под облаками доснимаются РСА за K итераций рандомизированного жадного
отбора кадров (правила выбора — те же, что в ``convergence.py``: greedy,
softmax, random). Запланированные кадры — это план наблюдений, а не физически
выполненные съёмки; вероятностные события реализации плана в данном исследовании
не моделируются.

Две метрики каждого запуска:

1. полнота досъёмки объектов, недоступных оптике:
   C_скрыт = area((∪ выбранных кадров) ∩ (∪ скрытых объектов)) / area(∪ скрытых объектов);
2. итоговая геометрическая полнота района после объединения оптического
   покрытия и досъёмки РСА:
   G = area((∪ доступных объектов) ∪ (∪ выбранных кадров) ∩ район) / area(района)
   (скобки неоднозначны: объединяются ∪ доступных объектов и (∪ кадров) ∩ район).

Для накопленной выборки первых m запусков (контрольные точки
``MonteCarloParams.checkpoints``) рассчитываются: среднее, стандартное
отклонение (ddof=1), стандартная ошибка среднего (SEM = s/√m) и 95%-й
доверительный интервал среднего (t-критическое из таблицы Стьюдента). По
фактическим данным проверяется: сходимость среднего (дрейф между контрольными
точками меньше заданной доли) и убывание статистической неопределённости
(ширина ДИ ∝ 1/√m; наклон на логарифмической шкале ≈ −0.5). Убывание
стандартного отклонения отдельных сценариев s не требуется и не проверяется —
s стремится к std генеральной совокупности, а не к нулю.

Минимальное число запусков для оценки средней полноты с точностью ±δ при
доверительной вероятности 95 %: n* = ⌈(z·s/δ)²⌉, z = 1.96, s — выборочное
std; дополнительно фиксируется первая контрольная точка, где фактическая
полуширина t-ДИ не превосходит δ.

Экспорт: ``monte_carlo_experiments.csv`` (итоги запусков),
``monte_carlo_checkpoints.csv`` (статистика по контрольным точкам),
``monte_carlo.json`` (всё исследование) и PNG-графики (панели по каждой метрике,
гистограммы, сравнение средних).

Важно: M — число независимых запусков Монте-Карло (новый сценарий каждый раз),
K — число итераций выбора кадров внутри одного запуска, число пролётов КА —
параметр кадров ``pass_azimuths_deg``; эти величины независимы.
"""

from __future__ import annotations

import heapq
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import __version__
from .config import ConfigError, ScenarioConfig
from .convergence import _T_CRIT_975, _softmax_pick, _t_crit_975
from .generator import generate_scenario

__all__ = [
    "MonteCarloParams",
    "run_monte_carlo_study", "run_monte_carlo_experiment",
    "compute_running_stats", "required_experiments_for_precision",
    "convergence_assessment",
    "write_monte_carlo_csv", "write_monte_carlo_json",
    "render_monte_carlo_charts",
    "run_cloud_sensitivity", "compute_combo_stats", "min_k_for_thresholds",
    "verify_cloud_model",
    "write_sensitivity_csv", "write_sensitivity_json",
    "render_sensitivity_charts",
    "t_crit", "classical_series", "required_trials",
    "k_sweep", "render_classical_charts",
]

METRIC_HIDDEN = "hidden_objects_coverage"
METRIC_GEOMETRIC = "final_geometric_completeness"
METRICS = (METRIC_GEOMETRIC, METRIC_HIDDEN)

_METRIC_TITLES_RU = {
    METRIC_GEOMETRIC: "Итоговая геометрическая полнота района G (оптика + РСА)",
    METRIC_HIDDEN: "Полнота досъёмки объектов, недоступных оптике, C",
}


@dataclass
class MonteCarloParams:
    """Регулируемые параметры исследования Монте-Карло.

    - ``experiments`` — M: число независимых запусков (≥ 2; каждый — новый
      сценарий из seed_start … seed_start + M − 1);
    - ``iterations`` — K: вычислительный бюджет алгоритма досъёмки (число
      итераций выбора кадров) в каждом запуске; одинаков во всех запусках;
    - ``checkpoints`` — контрольные точки m, в которых считается накопленная
      статистика (по возрастанию, не превосходят M);
    - ``seed_start`` — начальный seed сценариев;
    - ``cloud_fraction`` — вероятность того, что объект недоступен оптике
      (доля объектов под облачностью), 0…1;
    - ``selection`` / ``temperature`` — правило выбора кадра досъёмки
      (``greedy`` — детерминированный жадный с ленивой очередью, ``softmax`` —
      рандомизированный жадный, ``random`` — равномерный случайный);
    - ``precision`` — требуемая точность оценки средней полноты δ (доля);
    - ``confidence`` — доверительная вероятность (пока поддерживается 0.95).
    """

    experiments: int = 1000
    iterations: int = 70
    checkpoints: tuple[int, ...] = (10, 25, 50, 100, 200, 500, 1000)
    seed_start: int = 0
    cloud_fraction: float = 0.4
    selection: str = "greedy"
    temperature: float = 0.25
    precision: float = 0.01
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.experiments < 2:
            raise ValueError("monte_carlo.experiments: число запусков M должно быть не меньше 2 "
                             "(иначе стандартное отклонение и доверительный интервал не определены).")
        if self.iterations < 1:
            raise ValueError("monte_carlo.iterations: число итераций K должно быть не меньше 1.")
        cp = tuple(int(c) for c in self.checkpoints)
        if not cp or any(c < 2 for c in cp) or any(a >= b for a, b in zip(cp, cp[1:])):
            raise ValueError("monte_carlo.checkpoints: контрольные точки должны быть возрастающими "
                             "целыми, каждая не меньше 2.")
        if cp[-1] > self.experiments:
            raise ValueError("monte_carlo.checkpoints: последняя точка не может превосходить число запусков M.")
        object.__setattr__(self, "checkpoints", cp)
        if not (0.0 <= self.cloud_fraction <= 1.0):
            raise ValueError("monte_carlo.cloud_fraction: доля облачности должна быть от 0 до 1.")
        if self.selection not in ("softmax", "greedy", "random"):
            raise ValueError("monte_carlo.selection: допустимые значения «softmax», «greedy» и «random».")
        if self.temperature <= 0.0:
            raise ValueError("monte_carlo.temperature: температура должна быть положительной.")
        if self.precision <= 0.0:
            raise ValueError("monte_carlo.precision: требуемая точность должна быть положительной.")
        if not (0.0 < self.confidence < 1.0):
            raise ValueError("monte_carlo.confidence: доверительная вероятность должна быть в (0, 1).")


# ---------------------------------------------------------------------------
# Одиночный запуск Монте-Карло: новый сценарий + досъёмка РСА
# ---------------------------------------------------------------------------

def _cov_fraction(union: BaseGeometry, total: float) -> float:
    v = union.area / total
    return min(max(v, 0.0), 1.0)


def _combined_gain(
    idx: int,
    cand_district: list[BaseGeometry],
    cand_hidden: list[BaseGeometry] | None,
    area_district: list[float],
    area_hidden: list[float] | None,
    union_district: BaseGeometry,
    union_hidden: BaseGeometry | None,
    district_area: float,
    hidden_area: float,
) -> float:
    """Точный прирост комбинированной полноты от добавления кадра idx к объединениям.

    Комбинированная полнота = 0.5 · (доля покрытия района ∪ кадров + доля покрытия
    скрытых объектов ∪ кадров). Без скрытых объектов — только район.
    """
    inter_d = cand_district[idx].intersection(union_district).area if not union_district.is_empty else 0.0
    gain = (area_district[idx] - inter_d) / district_area
    if cand_hidden is not None and hidden_area > 0.0:
        inter_h = cand_hidden[idx].intersection(union_hidden).area if not union_hidden.is_empty else 0.0
        gain += (area_hidden[idx] - inter_h) / hidden_area
    return 0.5 * max(gain, 0.0)  # защита от численного шума


def _greedy_select(
    n_frames: int,
    params: MonteCarloParams,
    rng: np.random.Generator,
    cand_district: list[BaseGeometry],
    cand_hidden: list[BaseGeometry] | None,
    area_district: list[float],
    area_hidden: list[float] | None,
    district_area: float,
    hidden_area: float,
) -> list[int]:
    """Отобрать до K кадров рандомизированным жадным правилом (точный учёт перекрытий).

    greedy — ленивая очередь (lazy greedy): кадр хранится с верхней границей
    прироста; при извлечении граница пересчитывается точно, и кадр берётся,
    только если он всё ещё не хуже следующего кандидата. Для монотонной
    субмодулярной функции покрытия это даёт тот же выбор, что и наивный жадный
    перебор (классический результат Nemhauser–Wolsey–Fisher), но на порядок
    быстрее: точный прирост считается не K·N раз, а лишь для малой доли кадров.
    softmax и random перебирают все оставшиеся кадры на каждой итерации.
    """
    if params.selection == "random":
        order = list(range(n_frames))
        rng.shuffle(order)
        return order[: min(params.iterations, n_frames)]

    if params.selection == "softmax":
        remaining = list(range(n_frames))
        selected: list[int] = []
        union_d: BaseGeometry = GeometryCollection()
        union_h: BaseGeometry = GeometryCollection()
        for _ in range(min(params.iterations, n_frames)):
            gains = [
                _combined_gain(i, cand_district, cand_hidden, area_district, area_hidden,
                               union_d, union_h, district_area, hidden_area)
                for i in remaining
            ]
            pos = _softmax_pick(gains, params.temperature, rng)
            idx = remaining.pop(pos)
            selected.append(idx)
            union_d = cand_district[idx] if union_d.is_empty else union_d.union(cand_district[idx])
            if cand_hidden is not None:
                union_h = cand_hidden[idx] if union_h.is_empty else union_h.union(cand_hidden[idx])
        return selected

    # greedy с ленивой очередью: начальная граница — прирост к пустому объединению
    union_d: BaseGeometry = GeometryCollection()
    union_h: BaseGeometry = GeometryCollection()
    heap: list[tuple[float, int]] = []
    for i in range(n_frames):
        g = _combined_gain(i, cand_district, cand_hidden, area_district, area_hidden,
                           union_d, union_h, district_area, hidden_area)
        heap.append((-g, i))
    heapq.heapify(heap)
    selected = []
    while heap and len(selected) < params.iterations:
        neg_bound, idx = heapq.heappop(heap)
        g = _combined_gain(idx, cand_district, cand_hidden, area_district, area_hidden,
                           union_d, union_h, district_area, hidden_area)
        if not heap or g >= -heap[0][0] - 1e-15:
            selected.append(idx)
            union_d = cand_district[idx] if union_d.is_empty else union_d.union(cand_district[idx])
            if cand_hidden is not None:
                union_h = cand_hidden[idx] if union_h.is_empty else union_h.union(cand_hidden[idx])
        else:
            heapq.heappush(heap, (-g, idx))
    return selected


def run_monte_carlo_experiment(
    config: ScenarioConfig | dict,
    params: MonteCarloParams,
    scenario_seed: int,
    experiment_index: int = 0,
) -> dict:
    """Один запуск Монте-Карло: новый сценарий из scenario_seed + досъёмка РСА.

    Сценарий генерируется детерминированно из scenario_seed; маска оптической
    недоступности объектов — из независимого потока, привязанного к тому же seed
    (воспроизводимо и независимо между запусками). Возвращает только plain-данные:
    обе метрики, число скрытых объектов, итоговые площади и список кадров плана.
    """
    started = time.perf_counter()
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    # Размещение случайных сценариев из некоторых seed невозможно (не хватает
    # места). Такой сценарий детерминированно перегенерируется со следующим
    # seed; число повторов фиксируется. Это часть модели Монте-Карло: условное
    # распределение по реализуемым сценариям.
    from .placement import PlacementError

    scenario = None
    retries = 0
    seed_used = int(scenario_seed)
    for _ in range(100):
        try:
            config_dict["seed"] = seed_used
            scenario = generate_scenario(ScenarioConfig.model_validate(config_dict))
            break
        except PlacementError:
            seed_used += 1
            retries += 1
    if scenario is None:
        raise PlacementError(
            f"monte_carlo: не удалось сгенерировать реализуемый сценарий за 100 попыток "
            f"(стартовый seed {scenario_seed})."
        )

    district: BaseGeometry = scenario.district_local
    district_area = float(district.area)
    objects_union = unary_union([o.geometry for o in scenario.objects])

    # маска оптической недоступности: каждый объект независимо скрыт с вероятностью cloud_fraction
    mask_rng = np.random.default_rng(np.random.SeedSequence([seed_used, 20261126]))
    hidden_flags = mask_rng.random(len(scenario.objects)) < params.cloud_fraction
    hidden_geoms = [o.geometry for o, h in zip(scenario.objects, hidden_flags) if h]
    hidden_union = unary_union(hidden_geoms) if hidden_geoms else None
    hidden_area = float(hidden_union.area) if hidden_union is not None else 0.0
    optical_union = objects_union if hidden_union is None else objects_union.difference(hidden_union)

    cand_district = [f.geometry.intersection(district) for f in scenario.frames]
    area_district = [g.area for g in cand_district]
    if hidden_union is not None:
        cand_hidden = [f.geometry.intersection(hidden_union) for f in scenario.frames]
        area_hidden = [g.area for g in cand_hidden]
    else:
        cand_hidden = None
        area_hidden = None

    algo_rng = np.random.default_rng(np.random.SeedSequence([seed_used, 97]))
    selected = _greedy_select(
        len(scenario.frames), params, algo_rng,
        cand_district, cand_hidden, area_district, area_hidden,
        district_area, hidden_area,
    )

    union_frames_d: BaseGeometry = GeometryCollection()
    union_frames_h: BaseGeometry = GeometryCollection()
    for idx in selected:
        union_frames_d = cand_district[idx] if union_frames_d.is_empty else union_frames_d.union(cand_district[idx])
        if cand_hidden is not None:
            union_frames_h = cand_hidden[idx] if union_frames_h.is_empty else union_frames_h.union(cand_hidden[idx])

    final_union = optical_union if union_frames_d.is_empty else optical_union.union(union_frames_d)
    geometric = _cov_fraction(final_union, district_area)
    hidden_coverage = (_cov_fraction(union_frames_h, hidden_area)
                       if hidden_union is not None and hidden_area > 0.0 else math.nan)

    return {
        "experiment": experiment_index,
        "seed": seed_used,
        "placement_retries": retries,
        "n_objects": len(scenario.objects),
        "n_hidden": len(hidden_geoms),
        "n_candidates": len(scenario.frames),
        "district_area_m2": district_area,
        "objects_area_fraction": _cov_fraction(objects_union, district_area),
        "hidden_area_fraction": (hidden_area / district_area) if hidden_area > 0.0 else 0.0,
        "optical_district_coverage": _cov_fraction(optical_union, district_area),
        "hidden_objects_coverage": hidden_coverage,
        "final_geometric_completeness": geometric,
        "frames_selected": [scenario.frames[i].frame_id for i in selected],
        "duration_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Накопленная статистика по контрольным точкам
# ---------------------------------------------------------------------------

def _nan_stats(values: np.ndarray) -> dict:
    """Среднее, std (ddof=1), SEM и 95%-й t-ДИ по допустимым (не NaN) значениям."""
    valid = values[~np.isnan(values)]
    m = int(valid.size)
    if m < 2:
        return {"n_valid": m, "mean": math.nan, "std": math.nan, "sem": math.nan,
                "ci95_low": math.nan, "ci95_high": math.nan, "ci95_half": math.nan}
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    sem = std / math.sqrt(m)
    half = _t_crit_975(m - 1) * sem
    return {
        "n_valid": m,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half": half,
    }


def compute_running_stats(
    values: list[float],
    checkpoints: tuple[int, ...],
) -> list[dict]:
    """Накопленная статистика по первым m запускам для каждой контрольной точки m.

    NaN (например, запусков без скрытых объектов) исключаются; число
    допустимых значений сохраняется в ``n_valid``. Данные не сглаживаются.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("compute_running_stats: ожидается непустой одномерный массив значений.")
    stats: list[dict] = []
    for m in checkpoints:
        if m > arr.size:
            raise ValueError(f"compute_running_stats: контрольная точка m={m} превышает число запусков.")
        row = {"m": int(m), **_nan_stats(arr[:m])}
        stats.append(row)
    return stats


def required_experiments_for_precision(
    values: list[float],
    precision: float = 0.01,
    confidence: float = 0.95,
) -> dict:
    """Оценка минимального числа запусков для точности ±precision при данной вероятности.

    n* = ⌈(z·s/precision)²⌉ по выборочному std s полной выборки (нормальное
    приближение, z = 1.96 при 0.95 — при n* ≥ 30 разумно). Дополнительно —
    первая контрольная точка, где фактическая полуширина t-ДИ не превосходит
    precision (NaN, если такой нет).
    """
    if precision <= 0.0:
        raise ValueError("required_experiments_for_precision: точность должна быть положительной.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("required_experiments_for_precision: вероятность должна быть в (0, 1).")
    z = 1.96 if abs(confidence - 0.95) < 1e-12 else _z_norm(confidence)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("required_experiments_for_precision: нужно не менее 2 допустимых значений.")
    s = float(arr.std(ddof=1))
    n_star = math.ceil((z * s / precision) ** 2)
    return {
        "precision": precision,
        "confidence": confidence,
        "z": z,
        "std": s,
        "n_required": int(max(n_star, 2)),
        "n_sample": int(arr.size),
        "sufficient": bool(arr.size >= n_star),
    }


def _z_norm(confidence: float) -> float:
    """Квантиль стандартного нормального распределения (двусторонний ДИ), без scipy."""
    # рациональное приближение Абрамовица–Стигана 26.2.23 (|ε| < 4.5e-4)
    p = 1.0 - (1.0 - confidence) / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p if p > 0.5 else p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def convergence_assessment(
    running: dict[str, list[dict]],
    precision: float = 0.01,
) -> dict:
    """Проверка по фактическим данным: среднее сошлось, неопределённость убывает.

    Критерии (сходимость не предполагается заранее — возможен отрицательный вывод):
    - дрейф среднего: |mean(последняя точка) − mean(предыдущая)| < precision;
    - убывание неопределённости: наклон ln(ci_half) от ln(m) значимо близок
      к −0.5 (теория: SEM ∝ 1/√m); допуск ±0.15;
    - точность достигнута: полуширина t-ДИ на последней точке ≤ precision.
    """
    out: dict[str, dict] = {}
    for metric, rows in running.items():
        if len(rows) < 2:
            raise ValueError("convergence_assessment: нужно не менее двух контрольных точек.")
        ms = np.asarray([r["m"] for r in rows], dtype=float)
        halves = np.asarray([r["ci95_half"] for r in rows], dtype=float)
        means = np.asarray([r["mean"] for r in rows], dtype=float)
        drift = float(abs(means[-1] - means[-2]))
        slope = float(np.polyfit(np.log(ms), np.log(halves), 1)[0]) if np.all(halves > 0) else math.nan
        last = rows[-1]
        out[metric] = {
            "mean_final": last["mean"],
            "ci95_half_final": last["ci95_half"],
            "mean_drift_last": drift,
            "ci_half_loglog_slope": slope,
            "mean_converged": bool(drift < precision),
            "uncertainty_decreases": bool(not math.isnan(slope) and abs(slope + 0.5) <= 0.15),
            "precision_reached": bool(last["ci95_half"] <= precision),
            "std_note": "Убывание std отдельных сценариев не требуется: s → std генеральной совокупности.",
        }
    return out


# ---------------------------------------------------------------------------
# Оркестрация исследования
# ---------------------------------------------------------------------------

def run_monte_carlo_study(
    config: ScenarioConfig | dict,
    params: MonteCarloParams,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Выполнить M независимых запусков Монте-Карло: новый сценарий каждый раз.

    Параллелизм — отдельные процессы; результаты не зависят от числа workers.
    progress(done, total) вызывается по мере завершения запусков.
    """
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    if not config_dict.get("frames", {}).get("enabled"):
        raise ConfigError(
            "monte_carlo: исследование требует включённых кадров РСА "
            "(frames.enabled: true и непустые pass_azimuths_deg)."
        )
    ScenarioConfig.model_validate({**config_dict, "seed": config_dict.get("seed", 0)})
    if workers is None:
        workers = min(max(1, params.experiments), os.cpu_count() or 1)
    workers = max(1, min(workers, params.experiments))

    seeds = [params.seed_start + i for i in range(params.experiments)]
    experiments: list[dict] = []

    def _done() -> None:
        if progress:
            progress(len(experiments), params.experiments)

    if workers == 1:
        for i, s in enumerate(seeds):
            experiments.append(run_monte_carlo_experiment(config_dict, params, s, experiment_index=i))
            _done()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(run_monte_carlo_experiment, config_dict, params, s, experiment_index=i)
                for i, s in enumerate(seeds)
            ]
            for fut in as_completed(futures):
                experiments.append(fut.result())
                _done()

    experiments.sort(key=lambda e: e["experiment"])

    running: dict[str, list[dict]] = {}
    required: dict[str, dict] = {}
    for metric in METRICS:
        values = [e[metric] for e in experiments]
        running[metric] = compute_running_stats(values, params.checkpoints)
        required[metric] = required_experiments_for_precision(values, precision=params.precision,
                                                              confidence=params.confidence)

    warnings: list[str] = []
    n_retries = sum(e.get("placement_retries", 0) for e in experiments)
    if n_retries:
        warnings.append(
            f"{n_retries} повторов генерации сценария (размещение из исходного seed невозможно): "
            "использовано условное распределение по реализуемым сценариям."
        )
    n_nan_hidden = sum(1 for e in experiments if math.isnan(e["hidden_objects_coverage"]))
    if n_nan_hidden:
        warnings.append(
            f"{n_nan_hidden} запусков без объектов, недоступных оптике: метрика C_скрыт для них не определена "
            "и исключена из статистики."
        )
    if any(len(e["frames_selected"]) < params.iterations for e in experiments):
        warnings.append(
            "В некоторых запусках кадров-кандидатов меньше K: план досъёмки короче бюджета итераций."
        )

    assessment = convergence_assessment(running, precision=params.precision)

    return {
        "version": __version__,
        "mode": "monte_carlo_mean_convergence",
        "params": {**asdict(params), "checkpoints": list(params.checkpoints), "workers": workers},
        "scenario_config": {k: v for k, v in config_dict.items() if k != "seed"},
        "experiments": experiments,
        "running_stats": running,
        "required_experiments": required,
        "assessment": assessment,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Экспорт: CSV, JSON, PNG
# ---------------------------------------------------------------------------

def write_monte_carlo_csv(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """Записать CSV: итоги запусков и накопленную статистику по контрольным точкам."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    exp_path = out / "monte_carlo_experiments.csv"
    with exp_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment", "seed", "placement_retries", "n_objects", "n_hidden", "n_candidates",
            "objects_area_fraction", "hidden_area_fraction", "optical_district_coverage",
            "hidden_objects_coverage", "final_geometric_completeness",
            "frames_selected", "duration_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in study["experiments"]:
            writer.writerow({
                "experiment": e["experiment"],
                "seed": e["seed"],
                "placement_retries": e.get("placement_retries", 0),
                "n_objects": e["n_objects"],
                "n_hidden": e["n_hidden"],
                "n_candidates": e.get("n_candidates", ""),
                "objects_area_fraction": round(e.get("objects_area_fraction", math.nan), 8),
                "hidden_area_fraction": round(e.get("hidden_area_fraction", math.nan), 8),
                "optical_district_coverage": round(e["optical_district_coverage"], 8),
                "hidden_objects_coverage": (round(e["hidden_objects_coverage"], 8)
                                            if not math.isnan(e["hidden_objects_coverage"]) else ""),
                "final_geometric_completeness": round(e["final_geometric_completeness"], 8),
                "frames_selected": ";".join(e["frames_selected"]),
                "duration_s": round(e["duration_s"], 4),
            })
    paths["monte_carlo_experiments.csv"] = exp_path

    cp_path = out / "monte_carlo_checkpoints.csv"
    with cp_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["metric", "m", "n_valid", "mean", "std", "sem", "ci95_low", "ci95_high", "ci95_half"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            for row in study["running_stats"][metric]:
                writer.writerow({
                    "metric": metric,
                    **{k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()},
                })
    paths["monte_carlo_checkpoints.csv"] = cp_path
    return paths


def write_monte_carlo_json(study: dict, out_dir: str | Path) -> Path:
    """Записать всё исследование (параметры, запуски, статистика точек, оценки) в monte_carlo.json."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "monte_carlo.json"
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


_COLOR_REAL = "#4C78A8"
_COLOR_HIDDEN = "#E45756"
_COLOR_ALT = "#F58518"
_COLOR_TARGET = "green"


def render_monte_carlo_charts(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """Построить PNG-графики: панели сходимости по каждой метрике, гистограммы, сравнение средних."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    params = study["params"]
    precision = params["precision"]

    metric_values: dict[str, np.ndarray] = {
        m: np.asarray([e[m] for e in study["experiments"]], dtype=float) for m in METRICS
    }

    # 1–2) Панели по каждой метрике: среднее±ДИ, std, SEM, полуширина ДИ
    for metric in METRICS:
        rows = study["running_stats"][metric]
        m = np.asarray([r["m"] for r in rows], dtype=float)
        mean = np.asarray([r["mean"] for r in rows], dtype=float)
        lo = np.asarray([r["ci95_low"] for r in rows], dtype=float)
        hi = np.asarray([r["ci95_high"] for r in rows], dtype=float)
        std = np.asarray([r["std"] for r in rows], dtype=float)
        sem = np.asarray([r["sem"] for r in rows], dtype=float)
        half = np.asarray([r["ci95_half"] for r in rows], dtype=float)
        a = study["assessment"][metric]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax = axes[0][0]
        ax.fill_between(m, lo, hi, color=_COLOR_REAL, alpha=0.35, label="95%-й ДИ среднего")
        ax.plot(m, mean, color=_COLOR_REAL, marker="o", linewidth=1.8, label="накопленное среднее")
        ax.axhline(mean[-1], color=_COLOR_ALT, linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"среднее при M={int(m[-1])}: {mean[-1]:.4f}")
        ax.set_xscale("log")
        ax.set_title(f"Накопленное среднее (итог: {mean[-1]:.4f} ± {half[-1]:.4f})")
        ax.set_xlabel("число запусков M (лог. шкала)")
        ax.set_ylabel("средняя полнота")
        ax.legend(fontsize=8)

        ax = axes[0][1]
        ax.plot(m, std, color=_COLOR_ALT, marker="s", linewidth=1.8,
                label="std отдельных запусков (ddof=1)")
        ax.set_xscale("log")
        ax.set_title("Стандартное отклонение сценариев s(M)\n(не обязано убывать: s → std совокупности)")
        ax.set_xlabel("число запусков M (лог. шкала)")
        ax.set_ylabel("std")
        ax.legend(fontsize=8)

        ax = axes[1][0]
        ax.plot(m, sem, color=_COLOR_REAL, marker="^", linewidth=1.8, label="SEM = s/√M")
        ax.plot(m, sem[0] * np.sqrt(m[0] / m), color="gray", linestyle=":", linewidth=1.4,
                label="теория ∝ 1/√M")
        ax.set_xscale("log")
        if np.all(sem > 0):
            ax.set_yscale("log")
        ax.set_title("Стандартная ошибка среднего")
        ax.set_xlabel("число запусков M (лог. шкала)")
        ax.set_ylabel("SEM (лог. шкала)")
        ax.legend(fontsize=8)

        ax = axes[1][1]
        ax.plot(m, half, color=_COLOR_HIDDEN, marker="d", linewidth=1.8, label="полуширина 95%-го ДИ")
        ax.axhline(precision, color=_COLOR_TARGET, linestyle=":", linewidth=1.4,
                   label=f"требуемая точность ±{precision:.3g}")
        ax.set_xscale("log")
        ax.set_title(f"Статистическая неопределённость\n(наклон ln–ln {a['ci_half_loglog_slope']:.2f}, "
                     f"теория −0.5; точность {'достигнута' if a['precision_reached'] else 'не достигнута'})")
        ax.set_xlabel("число запусков M (лог. шкала)")
        ax.set_ylabel("полуширина ДИ")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"{_METRIC_TITLES_RU[metric]} — M = {int(m[-1])} запусков Монте-Карло, "
            f"K = {params['iterations']} итераций, досъёмка: {params['selection']}"
            + (f" T={params['temperature']}" if params["selection"] == "softmax" else "")
            + f", облачность {params['cloud_fraction']:.0%}",
            fontsize=11,
        )
        fig.text(0.01, 0.005,
                 "Кривые — фактические накопленные статистики по запускам; сглаживание не применяется. "
                 "M — число независимых запусков (новый сценарий), K — итерации выбора кадров.",
                 fontsize=7, alpha=0.7)
        fig.tight_layout(rect=(0, 0.015, 1, 0.96))
        p = out / f"monte_carlo_{metric}_panels.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths[p.name] = p

    # 3) Гистограммы итоговых полнот с средним ± ДИ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric in zip(axes, METRICS):
        vals = metric_values[metric]
        vals = vals[~np.isnan(vals)]
        last = study["running_stats"][metric][-1]
        # при почти константных значениях (реально для C при большом K) равномерные
        # бины не помещаются в диапазон данных — автоматический подбор, фолбэк к одному бину
        try:
            edges = np.histogram_bin_edges(vals, bins="auto")
        except ValueError:
            edges = 1
        ax.hist(vals, bins=edges, color=_COLOR_REAL, alpha=0.8, edgecolor="black", linewidth=0.4)
        ax.axvline(last["mean"], color=_COLOR_HIDDEN, linewidth=2.0,
                   label=f"среднее {last['mean']:.4f}")
        ax.axvspan(last["ci95_low"], last["ci95_high"], color=_COLOR_HIDDEN, alpha=0.25,
                   label=f"95%-й ДИ ±{last['ci95_half']:.4f}")
        ax.set_title(_METRIC_TITLES_RU[metric], fontsize=10)
        ax.set_xlabel("полнота")
        ax.set_ylabel("число запусков")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "monte_carlo_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # 4) Сравнение накопленных средних обеих метрик
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, color in ((METRIC_GEOMETRIC, _COLOR_REAL), (METRIC_HIDDEN, _COLOR_ALT)):
        rows = study["running_stats"][metric]
        m = [r["m"] for r in rows]
        mean = [r["mean"] for r in rows]
        lo = [r["ci95_low"] for r in rows]
        hi = [r["ci95_high"] for r in rows]
        ax.fill_between(m, lo, hi, color=color, alpha=0.25)
        ax.plot(m, mean, color=color, marker="o", linewidth=1.8,
                label=("итоговая геометрическая полнота G" if metric == METRIC_GEOMETRIC
                       else "полнота досъёмки скрытых объектов C"))
    ax.set_xscale("log")
    ax.set_title("Сходимость накопленных средних по числу запусков Монте-Карло")
    ax.set_xlabel("число запусков M (лог. шкала)")
    ax.set_ylabel("средняя полнота")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out / "monte_carlo_mean_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    return paths


# ---------------------------------------------------------------------------
# Сеточный анализ: облачность × число итераций K × правило выбора
# ---------------------------------------------------------------------------

def _combo_params(cloud_fraction: float, k: int, selection: str,
                  temperature: float = 0.25) -> MonteCarloParams:
    """Параметры одиночного запуска для ячейки сетки (experiments/checkpoints — формальность валидации)."""
    return MonteCarloParams(
        experiments=2, iterations=k, checkpoints=(2,),
        cloud_fraction=cloud_fraction, selection=selection, temperature=temperature,
    )


def _sensitivity_task(task):
    """Один запуск ячейки сетки (модульная функция — должна сериализоваться для пула)."""
    combo_i, p, k, sel, seed, m, config_dict, temperature = task
    params = _combo_params(p, k, sel, temperature)
    return combo_i, run_monte_carlo_experiment(config_dict, params, seed, experiment_index=m)


def run_cloud_sensitivity(
    config: ScenarioConfig | dict,
    cloud_fractions: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8),
    k_values: tuple[int, ...] = (5, 10, 20, 30, 50, 70),
    selections: tuple[str, ...] = ("greedy", "softmax"),
    experiments: int = 200,
    seed_start: int = 0,
    temperature: float = 0.25,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Сетка «облачность × K × правило»: не менее M запусков в каждой ячейке.

    Одинаковый диапазон seed во всех ячейках (seed_start … seed_start + M − 1):
    сценарии между ячейками спарены — различия оценок относятся к облачности,
    бюджету K и правилу выбора, а не к случайности сценариев. Каждая ячейка —
    M независимых запусков `run_monte_carlo_experiment` (новый сценарий каждый).
    Параллелизм — общий пул процессов по всем ячейкам; результаты не зависят
    от числа workers.
    """
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    if not config_dict.get("frames", {}).get("enabled"):
        raise ConfigError(
            "cloud_sensitivity: исследование требует включённых кадров РСА "
            "(frames.enabled: true и непустые pass_azimuths_deg)."
        )
    ScenarioConfig.model_validate({**config_dict, "seed": config_dict.get("seed", 0)})
    if experiments < 2:
        raise ValueError("cloud_sensitivity: число запусков M должно быть не меньше 2.")
    for p in cloud_fractions:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"cloud_sensitivity: облачность {p} вне диапазона [0, 1].")
    if any(k < 1 for k in k_values):
        raise ValueError("cloud_sensitivity: число итераций K должно быть не меньше 1.")
    for sel in selections:
        if sel not in ("softmax", "greedy", "random"):
            raise ValueError(f"cloud_sensitivity: недопустимое правило выбора «{sel}».")

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    combos = [(p, k, sel) for p in cloud_fractions for k in k_values for sel in selections]
    total = len(combos) * experiments
    tasks = [
        (combo_i, p, k, sel, seed_start + m, m, config_dict, temperature)
        for combo_i, (p, k, sel) in enumerate(combos)
        for m in range(experiments)
    ]

    results: dict[int, list[dict]] = {i: [] for i in range(len(combos))}
    done = 0

    if workers == 1:
        for task in tasks:
            combo_i, r = _sensitivity_task(task)
            results[combo_i].append(r)
            done += 1
            if progress:
                progress(done, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_sensitivity_task, t) for t in tasks]):
                combo_i, r = fut.result()
                results[combo_i].append(r)
                done += 1
                if progress:
                    progress(done, total)

    cells = []
    for combo_i, (p, k, sel) in enumerate(combos):
        runs = sorted(results[combo_i], key=lambda e: e["experiment"])
        cells.append({
            "cloud_fraction": p,
            "k": k,
            "selection": sel,
            "experiments": runs,
        })

    return {
        "version": __version__,
        "mode": "cloud_coverage_sensitivity",
        "grid": {
            "cloud_fractions": list(cloud_fractions),
            "k_values": list(k_values),
            "selections": list(selections),
            "temperature": temperature,
            "experiments": experiments,
            "seed_start": seed_start,
            "workers": workers,
        },
        "cells": cells,
    }


def compute_combo_stats(
    runs: list[dict],
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
    confidence: float = 0.95,
) -> dict:
    """Сводная статистика ячейки сетки: среднее/std/SEM/95 %-й ДИ метрик G и C
    и вероятность достижения порогов (доля запусков) с нормальным 95 %-м ДИ.

    Для вероятностей используется нормальный интервал Уилсона не применяется:
    при M ≥ 200 и p вдали от 0/1 np(1−p) ≥ 5 — нормальное приближение корректно;
    границы отсекаются к [0, 1].
    """
    if len(runs) < 2:
        raise ValueError("compute_combo_stats: нужно не менее 2 запусков.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("compute_combo_stats: confidence должно быть в (0, 1).")
    z = 1.96 if abs(confidence - 0.95) < 1e-12 else _z_norm(confidence)
    m = len(runs)

    def _metric_stats(name: str) -> dict:
        arr = np.asarray([r[name] for r in runs], dtype=float)
        valid = arr[~np.isnan(arr)]
        n = int(valid.size)
        if n < 2:
            return {"n_valid": n, "mean": math.nan, "std": math.nan, "sem": math.nan,
                    "ci95_half": math.nan}
        mean = float(valid.mean())
        std = float(valid.std(ddof=1))
        sem = std / math.sqrt(n)
        return {"n_valid": n, "mean": mean, "std": std, "sem": sem,
                "ci95_half": _t_crit_975(n - 1) * sem}

    stats = {"n_runs": m,
             "g": _metric_stats("final_geometric_completeness"),
             "c": _metric_stats("hidden_objects_coverage")}
    g = np.asarray([r["final_geometric_completeness"] for r in runs], dtype=float)
    c_arr = np.asarray([r["hidden_objects_coverage"] for r in runs], dtype=float)
    stats["thresholds"] = {}
    for gamma in thresholds:
        row = {}
        for key, arr in (("g", g), ("c", c_arr)):
            valid = arr[~np.isnan(arr)]
            n = int(valid.size)
            if n == 0:
                row[key] = {"threshold": gamma, "n_valid": 0, "share": math.nan,
                            "ci95_low": math.nan, "ci95_high": math.nan}
                continue
            share = float((valid >= gamma).mean())
            half = z * math.sqrt(share * (1.0 - share) / n)
            row[key] = {"threshold": gamma, "n_valid": n, "share": share,
                        "ci95_low": max(0.0, share - half), "ci95_high": min(1.0, share + half)}
        stats["thresholds"][gamma] = row
    return stats


def min_k_for_thresholds(
    study: dict,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
    confidence: float = 0.95,
) -> list[dict]:
    """Минимальное K достижения порога γ с вероятностью ≥ confidence для каждой
    пары (облачность, правило). Условие: доля запусков с метрикой ≥ γ не ниже
    confidence, причём нижняя граница её нормального 95 %-го ДИ тоже не ниже
    confidence − 0.05 (допуск на ошибку оценки вероятности); иначе k = None.
    """
    rows: list[dict] = []
    for cell in study["cells"]:
        stats = compute_combo_stats(cell["experiments"], thresholds=thresholds,
                                    confidence=confidence)
        cell["stats"] = stats
    for cell in study["cells"]:
        pass  # статистика уже присоединена выше
    by_combo: dict[tuple, dict[int, dict]] = {}
    for cell in study["cells"]:
        by_combo.setdefault((cell["cloud_fraction"], cell["selection"]), {})[cell["k"]] = cell
    for (p, sel), ks in sorted(by_combo.items()):
        for gamma in thresholds:
            k_min = None
            for k in sorted(ks):
                t = ks[k]["stats"]["thresholds"][gamma]["g"]
                if t["n_valid"] and t["share"] >= confidence and t["ci95_low"] >= confidence - 0.05:
                    k_min = k
                    break
            rows.append({"cloud_fraction": p, "selection": sel, "threshold": gamma,
                         "k_min": k_min, "confidence": confidence})
    return rows


def verify_cloud_model(study: dict) -> dict:
    """Проверка математической согласованности модели облачности и метрик.

    Для каждой ячейки: (1) оптическое покрытие должно совпадать с
    objects_area_fraction − hidden_area_fraction (равенство площадей,
    допуск на численный шум GEOS); (2) G ≥ оптического покрытия и G ≤ 1;
    (3) доля площади скрытых объектов ≈ cloud_fraction × доля площади всех
    объектов (независимость маски облачности от геометрии); (4) C ≤ 1.
    Возвращает словарь проверок с фактическими максимальными расхождениями.
    """
    checks = {
        "optical_equals_objects_minus_hidden": {"max_abs_diff": 0.0, "ok": True},
        "g_bounds": {"min_g": 1.0, "max_g": 0.0, "ok": True},
        "hidden_fraction_vs_cloud": {"max_abs_diff": 0.0, "ok": True},
        "c_bounds": {"min_c": 1.0, "max_c": 0.0, "ok": True},
    }
    p_ref = None
    for cell in study["cells"]:
        p_ref = cell["cloud_fraction"]
        for e in cell["experiments"]:
            d1 = abs(e["optical_district_coverage"]
                     - (e["objects_area_fraction"] - e["hidden_area_fraction"]))
            checks["optical_equals_objects_minus_hidden"]["max_abs_diff"] = max(
                checks["optical_equals_objects_minus_hidden"]["max_abs_diff"], d1)
            g = e["final_geometric_completeness"]
            checks["g_bounds"]["min_g"] = min(checks["g_bounds"]["min_g"], g)
            checks["g_bounds"]["max_g"] = max(checks["g_bounds"]["max_g"], g)
            if not (e["optical_district_coverage"] - 1e-9 <= g <= 1.0 + 1e-9):
                checks["g_bounds"]["ok"] = False
            if not math.isnan(e["hidden_objects_coverage"]):
                c = e["hidden_objects_coverage"]
                checks["c_bounds"]["min_c"] = min(checks["c_bounds"]["min_c"], c)
                checks["c_bounds"]["max_c"] = max(checks["c_bounds"]["max_c"], c)
                if not (-1e-9 <= c <= 1.0 + 1e-9):
                    checks["c_bounds"]["ok"] = False
        # проверка (3) — по средним ячейки; допуск — биномиальный масштаб:
        # при n объектов разброс доли скрытых ~ sqrt(p(1-p)/n), усреднение по
        # запускам делит на √M; для малых сценариев допуск автоматически шире
        objs = float(np.mean([e["objects_area_fraction"] for e in cell["experiments"]]))
        hid = float(np.mean([e["hidden_area_fraction"] for e in cell["experiments"]]))
        if objs > 0:
            n_obj = max(1, int(round(np.mean([e["n_objects"] for e in cell["experiments"]]))))
            n_runs = len(cell["experiments"])
            tol = max(0.05, 3.0 * math.sqrt(cell["cloud_fraction"]
                                            * (1.0 - cell["cloud_fraction"]) / n_obj)
                      / math.sqrt(n_runs))
            d3 = abs(hid / objs - cell["cloud_fraction"])
            checks["hidden_fraction_vs_cloud"]["max_abs_diff"] = max(
                checks["hidden_fraction_vs_cloud"]["max_abs_diff"], d3)
            checks["hidden_fraction_vs_cloud"]["tolerance"] = tol
    checks["optical_equals_objects_minus_hidden"]["ok"] = (
        checks["optical_equals_objects_minus_hidden"]["max_abs_diff"] < 1e-9)
    checks["g_bounds"]["ok"] = (checks["g_bounds"]["min_g"] >= 0.0
                                and checks["g_bounds"]["max_g"] <= 1.0)
    checks["hidden_fraction_vs_cloud"]["ok"] = (
        checks["hidden_fraction_vs_cloud"]["max_abs_diff"]
        < checks["hidden_fraction_vs_cloud"].get("tolerance", 0.05))
    checks["c_bounds"]["ok"] = (checks["c_bounds"]["min_c"] >= 0.0
                                and checks["c_bounds"]["max_c"] <= 1.0)
    return checks


def write_sensitivity_csv(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """CSV сеточного анализа: сводка по ячейкам и вероятности порогов."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for cell in study["cells"]:
        if "stats" not in cell:
            cell["stats"] = compute_combo_stats(cell["experiments"])

    summary_path = out / "cloud_sensitivity_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["cloud_fraction", "k", "selection", "n_runs",
                  "g_mean", "g_std", "g_sem", "g_ci95_half",
                  "c_mean", "c_std", "c_sem", "c_ci95_half"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cell in study["cells"]:
            s = cell["stats"]
            writer.writerow({
                "cloud_fraction": cell["cloud_fraction"],
                "k": cell["k"],
                "selection": cell["selection"],
                "n_runs": s["n_runs"],
                "g_mean": round(s["g"]["mean"], 8),
                "g_std": round(s["g"]["std"], 8),
                "g_sem": round(s["g"]["sem"], 8),
                "g_ci95_half": round(s["g"]["ci95_half"], 8),
                "c_mean": round(s["c"]["mean"], 8),
                "c_std": round(s["c"]["std"], 8),
                "c_sem": round(s["c"]["sem"], 8),
                "c_ci95_half": round(s["c"]["ci95_half"], 8),
            })
    paths["cloud_sensitivity_summary.csv"] = summary_path

    thr_path = out / "cloud_sensitivity_thresholds.csv"
    with thr_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["cloud_fraction", "k", "selection", "metric", "threshold",
                  "share", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cell in study["cells"]:
            for gamma, row in cell["stats"]["thresholds"].items():
                for metric in ("g", "c"):
                    t = row[metric]
                    writer.writerow({
                        "cloud_fraction": cell["cloud_fraction"],
                        "k": cell["k"],
                        "selection": cell["selection"],
                        "metric": "G" if metric == "g" else "C",
                        "threshold": gamma,
                        "share": (round(t["share"], 6) if t["n_valid"] else ""),
                        "ci95_low": (round(t["ci95_low"], 6) if t["n_valid"] else ""),
                        "ci95_high": (round(t["ci95_high"], 6) if t["n_valid"] else ""),
                    })
    paths["cloud_sensitivity_thresholds.csv"] = thr_path
    return paths


def _json_default(o):
    """Сериализация numpy-скаляров (bool_/float64/int64) в JSON."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(f"Объект типа {o.__class__.__name__} не сериализуется в JSON.")


def write_sensitivity_json(study: dict, out_dir: str | Path) -> Path:
    """Записать всё исследование сетки (ячейки, статистика, проверки) в cloud_coverage.json."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cloud_coverage.json"
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")
    return path


def render_sensitivity_charts(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """PNG-графики сетки: G(K) и C(K) по облачности, разброс, вероятности порогов."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for cell in study["cells"]:
        if "stats" not in cell:
            cell["stats"] = compute_combo_stats(cell["experiments"])

    clouds = sorted({cell["cloud_fraction"] for cell in study["cells"]})
    ks = sorted({cell["k"] for cell in study["cells"]})
    selections = sorted({cell["selection"] for cell in study["cells"]})
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, max(1, len(clouds))))

    def _cell(p, k, sel):
        for cell in study["cells"]:
            if cell["cloud_fraction"] == p and cell["k"] == k and cell["selection"] == sel:
                return cell
        return None

    # 1) G(K) и C(K) по облачности — панель на правило выбора
    for metric, key, title in (("G", "g", "Итоговая геометрическая полнота района G"),
                               ("C", "c", "Полнота досъёмки скрытых объектов C")):
        fig, axes = plt.subplots(1, len(selections), figsize=(7 * len(selections), 5.5),
                                 squeeze=False)
        for ax, sel in zip(axes[0], selections):
            for p, color in zip(clouds, colors):
                means, halves = [], []
                for k in ks:
                    s = _cell(p, k, sel)["stats"][key]
                    means.append(s["mean"])
                    halves.append(s["ci95_half"])
                means = np.asarray(means, dtype=float)
                halves = np.asarray(halves, dtype=float)
                ax.fill_between(ks, means - halves, means + halves, color=color, alpha=0.2)
                ax.plot(ks, means, color=color, marker="o", linewidth=1.8,
                        label=f"облачность {p:.0%}")
            ax.set_title(f"{sel}" + (" (T=0.25)" if sel == "softmax" else ""), fontsize=11)
            ax.set_xlabel("число итераций K")
            ax.set_ylabel(f"средняя полнота {metric}")
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title} vs бюджет K при разной облачности "
                     f"(M = {study['grid']['experiments']} запусков на точку; лента — 95 %-й ДИ)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out / f"cloud_sensitivity_{metric.lower()}_vs_k.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths[p.name] = p

    # 2) Межэкспериментальный разброс std(K) — G и C, жадный выбор
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (key, title) in zip(axes, (("g", "std G между запусками"),
                                       ("c", "std C между запусками"))):
        for p, color in zip(clouds, colors):
            stds = [_cell(p, k, "greedy")["stats"][key]["std"] for k in ks]
            ax.plot(ks, stds, color=color, marker="s", linewidth=1.8, label=f"облачность {p:.0%}")
        ax.set_title(f"{title} (greedy)", fontsize=11)
        ax.set_xlabel("число итераций K")
        ax.set_ylabel("std (ddof=1)")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Межэкспериментальный разброс: как сценарная случайность зависит от K и облачности", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "cloud_sensitivity_std_vs_k.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # 3) Вероятность достижения порогов P(G ≥ γ) vs K
    fig, axes = plt.subplots(1, len(selections), figsize=(7 * len(selections), 5.5),
                             squeeze=False)
    thresh_colors = {0.90: _COLOR_REAL, 0.95: _COLOR_ALT, 0.99: _COLOR_HIDDEN}
    for ax, sel in zip(axes[0], selections):
        for p, color in zip(clouds, colors):
            for gamma, tcolor in thresh_colors.items():
                shares = [_cell(p, k, sel)["stats"]["thresholds"][gamma]["g"]["share"] for k in ks]
                style = "-" if gamma == 0.90 else ("--" if gamma == 0.95 else ":")
                ax.plot(ks, shares, color=color, linestyle=style, marker="o", markersize=3.5,
                        linewidth=1.6, label=f"{p:.0%}, γ={gamma:.2f}")
        ax.axhline(0.95, color="black", linestyle="-.", linewidth=1.0, alpha=0.6,
                   label="confidence 0.95")
        ax.set_title(f"{sel}" + (" (T=0.25)" if sel == "softmax" else ""), fontsize=11)
        ax.set_xlabel("число итераций K")
        ax.set_ylabel("доля запусков с G ≥ γ")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Вероятность достижения порогов покрытия G (нормальный 95 %-й ДИ — в CSV)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "cloud_sensitivity_thresholds.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    return paths


# ---------------------------------------------------------------------------
# Классическая статистика (школа статистических испытаний):
# M*(n), D̃(n), σ̃(n), ε(β,n), потребный объём испытаний
# ---------------------------------------------------------------------------

# Квантили распределения Стьюдента для двустороннего ДИ с надёжностью β
# (уровни 0.95 / 0.975 / 0.995), df = 1…30; при df > 30 — нормальное приближение.
_T_CRIT = {
    0.90: (
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
        1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
        1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697,
    ),
    0.95: _T_CRIT_975,
    0.99: (
        63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169,
        3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
        2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750,
    ),
}

# Квантили стандартного нормального распределения (двусторонний ДИ).
_Z_NORM = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}

BETAS = (0.90, 0.95, 0.99)


def t_crit(beta: float, df: int) -> float:
    """Квантиль распределения Стьюдента для двустороннего ДИ с надёжностью β.

    Таблица для df = 1…30 (β ∈ {0.90, 0.95, 0.99}); при больших df —
    нормальное приближение z(β). Для прочих β — линейная интерполяция z.
    """
    if df < 1:
        return math.nan
    table = _T_CRIT.get(beta)
    if table is not None and df <= len(table):
        return table[df - 1]
    return _z_norm_two_sided(beta)


def _z_norm_two_sided(beta: float) -> float:
    if beta in _Z_NORM:
        return _Z_NORM[beta]
    return _z_norm(beta)


def classical_series(values: list[float], beta: float = 0.95) -> list[dict]:
    """Накопленные оценки по объёму испытаний n = 2…N (терминология классической
    школы статистических испытаний).

    Для каждого n: статистическое математическое ожидание M*(n), исправленная
    выборочная дисперсия D̃(n) (деление на n−1), оценка СКО σ̃(n) = √D̃,
    точность оценивания ε(β,n) = t(β,n−1)·σ̃/√n и доверительный интервал
    I = [M*−ε; M*+ε]. NaN-значения исключаются. Данные не сглаживаются.
    """
    if beta not in BETAS:
        raise ValueError(f"classical_series: надёжность β должна быть одной из {BETAS}.")
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n_total = int(arr.size)
    if n_total < 3:
        raise ValueError("classical_series: нужно не менее 3 допустимых значений.")
    series: list[dict] = []
    for n in range(2, n_total + 1):
        part = arr[:n]
        m_star = float(part.mean())
        d_tilde = float(((part - m_star) ** 2).sum() / (n - 1))
        sigma_tilde = math.sqrt(d_tilde)
        epsilon = t_crit(beta, n - 1) * sigma_tilde / math.sqrt(n)
        series.append({
            "n": n,
            "m_star": m_star,
            "d_tilde": d_tilde,
            "sigma_tilde": sigma_tilde,
            "epsilon": epsilon,
            "ci_low": m_star - epsilon,
            "ci_high": m_star + epsilon,
        })
    return series


def required_trials(sigma_tilde: float, epsilon0: float, beta: float = 0.95) -> dict:
    """Потребный объём испытаний: n ≥ (z(β)·σ̃/ε₀)².

    Предварительная нормальная оценка по пилотной выборке (σ̃ — её исправленное
    СКО). При малых n нормальная аппроксимация груба; значение возвращается
    вместе с флагом, и его нельзя представлять как строгую гарантию для
    произвольного распределения.
    """
    if sigma_tilde < 0.0:
        raise ValueError("required_trials: σ̃ не может быть отрицательным.")
    if epsilon0 <= 0.0:
        raise ValueError("required_trials: заданная точность ε₀ должна быть положительной.")
    if beta not in BETAS:
        raise ValueError(f"required_trials: надёжность β должна быть одной из {BETAS}.")
    z = _Z_NORM[beta]
    n_req = math.ceil((z * sigma_tilde / epsilon0) ** 2)
    return {
        "sigma_tilde": sigma_tilde,
        "epsilon0": epsilon0,
        "beta": beta,
        "z": z,
        "n_required": int(max(n_req, 3)),
        "normal_approximation": True,
        "note": "Пилотная нормальная оценка по σ̃ выборки; при малых n уточнять "
                "по распределению Стьюдента; не является строгой гарантией "
                "для произвольного распределения.",
    }


def render_classical_charts(
    series: list[dict],
    sweep: dict,
    beta: float,
    epsilon0: float,
    out_dir: str | Path,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, Path]:
    """Построить пять обязательных графиков статистического исследования.

    1 — статистическое математическое ожидание M*(n) с доверительными границами;
    2 — исправленная выборочная дисперсия D̃(n);
    3 — оценка среднего квадратического отклонения σ̃(n);
    4 — точность оценивания ε(β,n) с линией заданной точности ε₀;
    5 — сходимость алгоритма досъёмки РСА: средние G(K) и C(K) по n испытаниям
    с доверительными границами при надёжности β и порогами 90/95/99 %.

    Графики 1–4 — сходимость статистической оценки при росте объёма испытаний n
    (фиксированный K); график 5 — сходимость алгоритма планирования при росте
    числа итераций K (фиксированный объём испытаний). Это разные процессы.
    ``sweep`` — результат `k_sweep`: dict K -> {"g": {...}, "c": {...}} со
    средними и полуширинами ДИ при выбранной β.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    n = np.asarray([r["n"] for r in series], dtype=float)
    m_star = np.asarray([r["m_star"] for r in series], dtype=float)
    lo = np.asarray([r["ci_low"] for r in series], dtype=float)
    hi = np.asarray([r["ci_high"] for r in series], dtype=float)
    d_tilde = np.asarray([r["d_tilde"] for r in series], dtype=float)
    sigma = np.asarray([r["sigma_tilde"] for r in series], dtype=float)
    eps = np.asarray([r["epsilon"] for r in series], dtype=float)

    # График 1. Статистическое математическое ожидание M*(n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(n, lo, hi, color=_COLOR_REAL, alpha=0.3,
                    label=f"доверительные границы I(β={beta:.2f}, n)")
    ax.plot(n, m_star, color=_COLOR_REAL, linewidth=1.8, label="M*(n) — статистическое мат. ожидание")
    ax.axhline(m_star[-1], color=_COLOR_ALT, linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"M*({int(n[-1])}) = {m_star[-1]:.4f}")
    ax.set_title("График 1. Статистическое математическое ожидание M*(n)")
    ax.set_xlabel("объём статистических испытаний n")
    ax.set_ylabel("M*(n) — геометрическая полнота G")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out / "stat_m_star.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # График 2. Исправленная выборочная дисперсия D̃(n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n, d_tilde, color=_COLOR_REAL, linewidth=1.8,
            label="D̃(n) = Σ(xᵢ−M*)²/(n−1)")
    ax.axhline(d_tilde[-1], color=_COLOR_ALT, linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"D̃({int(n[-1])}) = {d_tilde[-1]:.3g}")
    ax.set_title("График 2. Исправленная выборочная дисперсия D̃(n)")
    ax.set_xlabel("объём статистических испытаний n")
    ax.set_ylabel("D̃(n)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "stat_d_tilde.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # График 3. Оценка среднего квадратического отклонения σ̃(n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n, sigma, color=_COLOR_REAL, linewidth=1.8, label="σ̃(n) = √D̃(n)")
    ax.axhline(sigma[-1], color=_COLOR_ALT, linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"σ̃({int(n[-1])}) = {sigma[-1]:.4f}")
    ax.set_title("График 3. Оценка среднего квадратического отклонения σ̃(n)")
    ax.set_xlabel("объём статистических испытаний n")
    ax.set_ylabel("σ̃(n)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "stat_sigma_tilde.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # График 4. Точность оценивания ε(β,n) и линия ε₀
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n, eps, color=_COLOR_REAL, linewidth=1.8,
            label=f"ε(β={beta:.2f}, n) = t(β,n−1)·σ̃(n)/√n")
    ax.axhline(epsilon0, color=_COLOR_HIDDEN, linestyle=":", linewidth=1.6,
               label=f"заданная точность ε₀ = {epsilon0:.4f}")
    below = eps <= epsilon0
    if below.any():
        n_first = int(n[np.nonzero(below)[0][0]])
        ax.axvline(n_first, color=_COLOR_ALT, linestyle="--", linewidth=1.2,
                   label=f"ε ≤ ε₀ начиная с n = {n_first}")
    ax.set_title("График 4. Точность оценивания ε(β,n)")
    ax.set_xlabel("объём статистических испытаний n")
    ax.set_ylabel("ε(β, n)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "stat_epsilon.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # График 5. Сходимость алгоритма досъёмки РСА: G(K), C(K)
    ks = sorted(sweep)
    g_mean = np.asarray([sweep[k]["g"]["mean"] for k in ks], dtype=float)
    g_half = np.asarray([sweep[k]["g"]["ci_half"] for k in ks], dtype=float)
    c_mean = np.asarray([sweep[k]["c"]["mean"] for k in ks], dtype=float)
    c_half = np.asarray([sweep[k]["c"]["ci_half"] for k in ks], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(ks, g_mean - g_half, g_mean + g_half, color=_COLOR_REAL, alpha=0.25)
    ax.plot(ks, g_mean, color=_COLOR_REAL, marker="o", linewidth=2.0,
            label="M*[G(K)] — полнота района (оптика + РСА)")
    ax.fill_between(ks, np.maximum(c_mean - c_half, 0.0), np.minimum(c_mean + c_half, 1.0),
                    color=_COLOR_ALT, alpha=0.2)
    ax.plot(ks, c_mean, color=_COLOR_ALT, marker="s", linewidth=1.8, linestyle="--",
            label="M*[C(K)] — полнота досъёмки скрытых объектов")
    thresh_colors = {0.90: "green", 0.95: _COLOR_HIDDEN, 0.99: "purple"}
    for gamma in thresholds:
        ax.axhline(gamma, color=thresh_colors.get(gamma, "gray"), linestyle=":", linewidth=1.2,
                   label=f"порог {gamma:.0%}")
        hit = np.nonzero(g_mean >= gamma)[0]
        if hit.size:
            ax.annotate(f"K={ks[hit[0]]}", (ks[hit[0]], gamma), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=thresh_colors.get(gamma, "gray"))
    ax.set_title(f"График 5. Сходимость алгоритма досъёмки РСА "
                 f"(n испытаний на точку, доверительные границы при β={beta:.2f})")
    ax.set_xlabel("число итераций алгоритма K (не число пролётов КА)")
    ax.set_ylabel("геометрическая полнота, %")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out / "stat_convergence_k.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    return paths


def k_sweep(
    config: ScenarioConfig | dict,
    cloud_fraction: float,
    k_values: tuple[int, ...],
    selection: str,
    experiments: int,
    seed_start: int = 0,
    temperature: float = 0.25,
    beta: float = 0.95,
    workers: int | None = None,
    keep_runs: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Развеска по K: для каждого K — n независимых испытаний, средние G и C
    с доверительными полуширинами при надёжности β (по распределению Стьюдента).

    Те же seed во всех точках (спаренные сценарии). Возвращает dict
    K -> {"g": {"mean", "ci_half", "std", "n"}, "c": {...}}; при keep_runs=True
    каждая точка дополнительно содержит "runs" — сырые итоги испытаний
    (для накопленных оценок M*(n), D̃(n), σ̃(n), ε(β,n) при фиксированном K).
    """
    if beta not in BETAS:
        raise ValueError(f"k_sweep: надёжность β должна быть одной из {BETAS}.")
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    if not config_dict.get("frames", {}).get("enabled"):
        raise ConfigError(
            "k_sweep: исследование требует включённых кадров РСА "
            "(frames.enabled: true и непустые pass_azimuths_deg)."
        )
    ScenarioConfig.model_validate({**config_dict, "seed": config_dict.get("seed", 0)})
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    total = len(k_values) * experiments
    tasks = [
        (k, seed_start + m, m, config_dict, cloud_fraction, selection, temperature)
        for k in k_values
        for m in range(experiments)
    ]
    results: dict[int, list[dict]] = {k: [] for k in k_values}
    done = 0

    if workers == 1:
        for t in tasks:
            k, r = _k_sweep_task(t)
            results[k].append(r)
            done += 1
            if progress:
                progress(done, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_k_sweep_task, t) for t in tasks]):
                k, r = fut.result()
                results[k].append(r)
                done += 1
                if progress:
                    progress(done, total)

    sweep: dict[int, dict] = {}
    for k in k_values:
        runs = sorted(results[k], key=lambda e: e["experiment"])
        row: dict[str, dict] = {}
        for key, name in (("g", "final_geometric_completeness"), ("c", "hidden_objects_coverage")):
            arr = np.asarray([e[name] for e in runs], dtype=float)
            arr = arr[~np.isnan(arr)]
            nn = int(arr.size)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if nn >= 2 else math.nan
            half = (t_crit(beta, nn - 1) * std / math.sqrt(nn)) if nn >= 2 else math.nan
            row[key] = {"mean": mean, "std": std, "ci_half": half, "n": nn,
                        "epsilon": half}
        if keep_runs:
            row["runs"] = runs
        sweep[k] = row
    return sweep


def _k_sweep_task(task):
    """Один испытание развески по K (модульная функция — для пула процессов)."""
    k, seed, m, config_dict, cloud_fraction, selection, temperature = task
    params = _combo_params(cloud_fraction, k, selection, temperature)
    return k, run_monte_carlo_experiment(config_dict, params, seed, experiment_index=m)
