"""Статистическое исследование сходимости итерационного алгоритма формирования кадров РСА.

Исследуемый алгоритм — рандомизированный жадный отбор кадров-кандидатов. На каждой
итерации k для каждого ещё не выбранного кадра вычисляется точный прирост полноты
покрытия при добавлении его к объединению уже выбранных кадров (перекрытия в
объединении не считаются дважды), и один кадр выбирается по одному из правил
(``ConvergenceParams.selection``):

- ``softmax`` — взвешенный случайный выбор с температурой: вес кадра
  ∝ exp(ΔI / (T·ΔI_max)); при T → 0 алгоритм становится чисто жадным,
  при T → ∞ — равномерно случайным (правило по умолчанию);
- ``greedy`` — детерминированный выбор кадра с максимальным приростом
  (при равенстве — первый по порядку; RNG не используется, все эксперименты
  на одном сценарии дают одинаковую траекторию);
- ``random`` — равномерный случайный выбор из оставшихся кадров (RNG, приросты
  не вычисляются — кадры не ранжируются).

Полнота информации после k итераций:

    I(k) = 0.5 · (S_район(k) + S_объекты(k)),
    S_район(k)    = area((∪ выбранных кадров) ∩ район) / area(района),
    S_объекты(k)  = area((∪ выбранных кадров) ∩ (∪ объектов)) / area(∪ объектов).

Для фиксированного сценария (одинаковые район, объекты и кадры-кандидаты во всех
экспериментах) проводится M независимых экспериментов с seed = seed_start …
seed_start + M − 1, каждый содержит K последовательных итераций. Случайность —
только в работе алгоритма: район и объекты генерируются детерминированно из seed
сценария и не перегенерируются; неизменность сценария контролируется отпечатком
в каждом эксперименте. Для каждого эксперимента сохраняется полная траектория I(k).

Статистика по номеру итерации k (по результатам независимых экспериментов):
среднее, стандартное отклонение (ddof=1), медиана, 10-й и 90-й процентили,
95%-й доверительный интервал среднего (t-критическое из встроенной таблицы
распределения Стьюдента, без scipy) и средний прирост I(k) − I(k−1).

Данные на графиках не сглаживаются и полоса разброса искусственно не сужается:
на каждой итерации показываются фактические p10–p90 и фактическое std по M
экспериментам.

Диагностика гипотезы о сходимости по фактическим результатам:

- ``converged`` — средний прирост стабилизировался (скользящее окно среднего
  прироста не превосходит tolerance) и итоговая средняя полнота не ниже
  минимально требуемой;
- ``stagnation`` — прирост стабилизировался, но итоговая полнота ниже требуемой;
- ``different_local_optima`` — прирост стабилизировался, однако разброс итоговых
  полнот между экспериментами выше порога: эксперименты пришли к разным
  локальным решениям;
- ``no_convergence`` — за K итераций критерий стабилизации не выполнен.

Экспорт: ``convergence_trajectories.csv`` (полная траектория каждого эксперимента),
``convergence_summary.csv`` (статистика по номерам итераций),
``convergence_experiments.csv`` (итоги экспериментов), ``convergence.json``
(всё исследование) и четыре PNG-графика.
"""

from __future__ import annotations

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
from .generator import generate_scenario

__all__ = [
    "ConvergenceParams",
    "run_convergence_study", "run_experiment",
    "compute_iteration_stats", "diagnose_convergence",
    "coverage_threshold_analysis", "std_trend_analysis",
    "write_convergence_csv", "write_convergence_json",
    "render_convergence_charts", "render_comparison_charts",
]

# t-критические значения уровня 0.975 распределения Стьюдента для df = 1…30
# (двусторонний 95%-й ДИ среднего); при df > 30 — нормальное приближение 1.96.
_T_CRIT_975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _t_crit_975(df: int) -> float:
    """t-критическое уровня 0.975 для df степеней свободы (df ≥ 1)."""
    if df < 1:
        return math.nan
    if df <= len(_T_CRIT_975):
        return _T_CRIT_975[df - 1]
    return 1.96


@dataclass
class ConvergenceParams:
    """Регулируемые параметры исследования сходимости.

    - ``experiments`` — M: число независимых экспериментов (≥ 2, иначе std и ДИ
      не определены);
    - ``iterations`` — K: число последовательных итераций алгоритма в эксперименте;
    - ``seed_start`` — начальный seed: эксперимент m использует seed_start + m;
    - ``stabilization_tol`` / ``stabilization_window`` — критерий стабилизации:
      стабилизировалось на итерации k, если средний прирост по окну из
      ``stabilization_window`` итераций, заканчивающемуся в k, не превосходит
      ``stabilization_tol`` и это выполняется до конца K;
    - ``min_completeness`` — минимально требуемая полнота информации;
    - ``temperature`` — температура softmax рандомизированного жадного выбора
      (случайность исследуемого алгоритма; используется только при
      ``selection = "softmax"``);
    - ``selection`` — правило выбора кадра: ``"softmax"`` (по умолчанию),
      ``"greedy"`` (детерминированный жадный) или ``"random"`` (равномерный
      случайный без ранжирования приростов);
    - ``spread_tol`` — порог итогового std между экспериментами для диагностики
      разных локальных решений.
    """

    experiments: int = 20
    iterations: int = 30
    seed_start: int = 0
    stabilization_tol: float = 1e-3
    stabilization_window: int = 3
    min_completeness: float = 0.9
    temperature: float = 0.25
    selection: str = "softmax"
    spread_tol: float = 0.02

    def __post_init__(self) -> None:
        if self.experiments < 2:
            raise ValueError("convergence.experiments: число экспериментов M должно быть не меньше 2 "
                             "(иначе стандартное отклонение и доверительный интервал не определены).")
        if self.iterations < 1:
            raise ValueError("convergence.iterations: число итераций K должно быть не меньше 1.")
        if self.stabilization_tol <= 0.0:
            raise ValueError("convergence.stabilization_tol: порог стабилизации должен быть положительным.")
        if self.stabilization_window < 1:
            raise ValueError("convergence.stabilization_window: окно стабилизации должно быть не меньше 1.")
        if not (0.0 <= self.min_completeness <= 1.0):
            raise ValueError("convergence.min_completeness: полнота информации — доля от 0 до 1.")
        if self.temperature <= 0.0:
            raise ValueError("convergence.temperature: температура должна быть положительной.")
        if self.selection not in ("softmax", "greedy", "random"):
            raise ValueError("convergence.selection: допустимые значения «softmax», «greedy» и «random».")
        if self.spread_tol < 0.0:
            raise ValueError("convergence.spread_tol: порог разброса не может быть отрицательным.")


# ---------------------------------------------------------------------------
# Итерационный алгоритм формирования кадров и одиночный эксперимент
# ---------------------------------------------------------------------------

def _softmax_pick(gains: list[float], temperature: float, rng: np.random.Generator) -> int:
    """Выбрать позицию в списке по softmax весам приростов (стабильный к переполнению).

    Приросты нормируются на максимум: вес ∝ exp((g − g_max) / (T·g_max)).
    Все приросты ≤ 0 (нет улучшения) — равномерный выбор среди оставшихся.
    """
    g = np.asarray(gains, dtype=float)
    g_max = float(g.max()) if g.size else 0.0
    if g_max <= 0.0 or g.size == 0:
        return int(rng.integers(g.size)) if g.size else 0
    scale = max(temperature * g_max, 1e-15)
    w = np.exp((g - g_max) / scale)
    p = w / w.sum()
    return int(rng.choice(g.size, p=p))


def _scenario_fingerprint(scenario) -> dict:
    """Отпечаток фиксированного сценария: контроль неизменности района между экспериментами."""
    first = (0.0, 0.0)
    if scenario.objects:
        first = scenario.objects[0].geometry.exterior.coords[0]
    return {
        "objects_count": len(scenario.objects),
        "frames_count": len(scenario.frames),
        "district_area_m2": round(float(scenario.district_local.area), 3),
        "objects_area_m2": round(float(sum(o.area_m2 for o in scenario.objects)), 3),
        "first_vertex_x": round(float(first[0]), 3),
        "first_vertex_y": round(float(first[1]), 3),
    }


def _cov_fraction(union: BaseGeometry, total: float) -> float:
    """Доля покрытия с отсечением машинного шума пересечений/объединений к диапазону [0, 1]."""
    v = union.area / total
    return min(max(v, 0.0), 1.0)


def run_experiment(
    config: ScenarioConfig | dict,
    params: ConvergenceParams,
    algo_seed: int,
    experiment_index: int = 0,
) -> dict:
    """Один эксперимент: фиксированный сценарий + K итераций алгоритма с RNG(algo_seed).

    Район, объекты и кадры-кандидаты генерируются детерминированно из seed сценария
    и одинаковы во всех экспериментах; algo_seed влияет только на выбор кадров.
    Возвращает только plain-данные (без shapely-геометрий), включая полную
    траекторию I(k), обе компоненты полноты и список выбранных кадров.
    """
    started = time.perf_counter()
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    scenario = generate_scenario(ScenarioConfig.model_validate(config_dict))

    district: BaseGeometry = scenario.district_local
    district_area = float(district.area)
    objects_union = unary_union([o.geometry for o in scenario.objects])
    objects_area = float(objects_union.area)

    # Точный учёт покрытия без повторного счёта перекрытий: поддерживаем два
    # объединения — пересечения выбранных кадров с районом и с объектами;
    # (∪ кадров) ∩ D = ∪ (кадр ∩ D), поэтому наращивание объединения точно.
    cand_district = [f.geometry.intersection(district) for f in scenario.frames]
    cand_objects = [f.geometry.intersection(objects_union) for f in scenario.frames]
    area_district = [g.area for g in cand_district]
    area_objects = [g.area for g in cand_objects]

    rng = np.random.default_rng(algo_seed)
    remaining = list(range(len(scenario.frames)))
    union_district: BaseGeometry = GeometryCollection()
    union_objects: BaseGeometry = GeometryCollection()
    current = 0.0

    trajectory: list[float] = []
    district_cov: list[float] = []
    objects_cov: list[float] = []
    frames_selected: list[str] = []
    marginal_gains: list[float] = []

    for _ in range(params.iterations):
        if not remaining:  # кадры исчерпаны: полнота не меняется, итерации досчитываются
            trajectory.append(current)
            district_cov.append(_cov_fraction(union_district, district_area))
            objects_cov.append(_cov_fraction(union_objects, objects_area))
            frames_selected.append("")
            marginal_gains.append(0.0)
            continue

        if params.selection == "random":
            # равномерный случайный выбор без ранжирования приростов
            pos = int(rng.integers(len(remaining)))
        else:
            gains: list[float] = []
            for idx in remaining:
                inter_d = cand_district[idx].intersection(union_district).area if not union_district.is_empty else 0.0
                inter_o = cand_objects[idx].intersection(union_objects).area if not union_objects.is_empty else 0.0
                gain = 0.5 * (
                    (area_district[idx] - inter_d) / district_area
                    + (area_objects[idx] - inter_o) / objects_area
                )
                gains.append(max(gain, 0.0))  # защита от численного шума
            if params.selection == "greedy":
                pos = int(np.argmax(np.asarray(gains)))  # при равенстве — первый по порядку (детерминированно)
            else:
                pos = _softmax_pick(gains, params.temperature, rng)

        idx = remaining.pop(pos)
        previous = current
        union_district = union_district.union(cand_district[idx]) if not union_district.is_empty else cand_district[idx]
        union_objects = union_objects.union(cand_objects[idx]) if not union_objects.is_empty else cand_objects[idx]
        current = 0.5 * (_cov_fraction(union_district, district_area) + _cov_fraction(union_objects, objects_area))

        trajectory.append(current)
        district_cov.append(_cov_fraction(union_district, district_area))
        objects_cov.append(_cov_fraction(union_objects, objects_area))
        frames_selected.append(scenario.frames[idx].frame_id)
        marginal_gains.append(current - previous)

    return {
        "experiment": experiment_index,
        "seed": int(algo_seed),
        "trajectory": trajectory,
        "district_coverage": district_cov,
        "objects_coverage": objects_cov,
        "frames_selected": frames_selected,
        "marginal_gains": marginal_gains,
        "final_completeness": trajectory[-1] if trajectory else 0.0,
        "duration_s": time.perf_counter() - started,
        "scenario_fingerprint": _scenario_fingerprint(scenario),
    }


# ---------------------------------------------------------------------------
# Статистика по номерам итераций
# ---------------------------------------------------------------------------

def compute_iteration_stats(trajectories: list[list[float]]) -> list[dict]:
    """Статистика по каждому номеру итерации k = 1…K по M независимым экспериментам.

    Для каждого k: среднее, стандартное отклонение (ddof=1), медиана, 10-й и 90-й
    процентили (линейная интерполяция numpy), 95%-й доверительный интервал среднего
    (t-критическое) и средний прирост I(k) − I(k−1) (I(0) = 0). При M = 1 std и
    ДИ — NaN. Данные не сглаживаются.
    """
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("compute_iteration_stats: ожидается непустой массив траекторий (M, K).")
    m, k_count = arr.shape
    t_crit = _t_crit_975(m - 1) if m >= 2 else math.nan
    stats: list[dict] = []
    for k in range(k_count):
        col = arr[:, k]
        mean = float(col.mean())
        std = float(col.std(ddof=1)) if m >= 2 else math.nan
        p10, median, p90 = (float(v) for v in np.percentile(col, [10.0, 50.0, 90.0]))
        if m >= 2:
            half = t_crit * std / math.sqrt(m)
            ci_low, ci_high = mean - half, mean + half
        else:
            ci_low = ci_high = math.nan
        prev = arr[:, k - 1] if k > 0 else np.zeros(m)
        mean_increment = float((col - prev).mean())
        stats.append({
            "iteration": k + 1,
            "mean": mean,
            "std": std,
            "median": median,
            "p10": p10,
            "p90": p90,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "mean_increment": mean_increment,
        })
    return stats


# ---------------------------------------------------------------------------
# Диагностика сходимости
# ---------------------------------------------------------------------------

_VERDICTS_RU = {
    "converged": "Сходимость подтверждена: прирост стабилизировался, итоговая полнота не ниже требуемой",
    "stagnation": "Стагнация: прирост стабилизировался ниже требуемой полноты",
    "different_local_optima": "Эксперименты сошлись к разным локальным решениям",
    "no_convergence": "Сходимость не достигнута за K итераций",
}


def diagnose_convergence(iteration_stats: list[dict], params: ConvergenceParams) -> dict:
    """Проверить гипотезу о сходимости по фактическим результатам.

    Стабилизация: начиная с итерации k средний прирост по скользящему окну длины
    ``stabilization_window`` не превосходит ``stabilization_tol``. Диагнозы:
    «converged», «stagnation», «different_local_optima» (итоговый std между
    экспериментами выше ``spread_tol``), «no_convergence». Случаи стагнации и
    разных локальных решений фиксируются явно, данные не сглаживаются.
    """
    if not iteration_stats:
        raise ValueError("diagnose_convergence: пустая статистика по итерациям.")
    k_total = len(iteration_stats)
    increments = [s["mean_increment"] for s in iteration_stats]
    window = min(params.stabilization_window, k_total)

    def _window_mean(k0: int) -> float:
        start = max(0, k0 - window + 1)
        return float(np.mean(increments[start:k0 + 1]))

    stab_flags = [_window_mean(k0) <= params.stabilization_tol for k0 in range(k_total)]
    stabilized_at: int | None = None
    for k0 in range(k_total):
        if all(stab_flags[k0:]):
            stabilized_at = k0 + 1
            break

    final = iteration_stats[-1]
    final_mean = float(final["mean"])
    final_std = final["std"]
    local_optima = bool(not math.isnan(final_std) and final_std > params.spread_tol)

    if stabilized_at is None:
        verdict = "no_convergence"
    elif local_optima:
        verdict = "different_local_optima"
    elif final_mean >= params.min_completeness:
        verdict = "converged"
    else:
        verdict = "stagnation"

    band = float(final["p90"] - final["p10"])
    if stabilized_at is None:
        detail = (
            f"За {k_total} итераций средний прирост по окну {window} не опустился "
            f"постоянно до порога {params.stabilization_tol:.3g}: сходимость не подтверждена. "
            f"Итоговая средняя полнота {final_mean:.4f} (требуемая {params.min_completeness:.3g}), "
            f"std между экспериментами {final_std:.4f}."
        )
    else:
        detail = (
            f"Стабилизация на итерации {stabilized_at} из {k_total} (окно {window}, "
            f"порог прироста {params.stabilization_tol:.3g}). Итоговая средняя полнота "
            f"{final_mean:.4f} при требуемой {params.min_completeness:.3g}; std между экспериментами "
            f"{final_std:.4f} (порог {params.spread_tol:.3g}), разброс p10–p90 {band:.4f}."
        )

    return {
        "stabilized": stabilized_at is not None,
        "stabilized_at_iteration": stabilized_at,
        "iterations_total": k_total,
        "final_mean": final_mean,
        "final_std": final_std,
        "final_median": float(final["median"]),
        "final_p10": float(final["p10"]),
        "final_p90": float(final["p90"]),
        "final_band_p10_p90": band,
        "required_completeness": params.min_completeness,
        "spread_tol": params.spread_tol,
        "local_optima_detected": local_optima,
        "verdict": verdict,
        "verdict_ru": _VERDICTS_RU[verdict],
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Анализ достижимости порогов покрытия и тренда разброса
# ---------------------------------------------------------------------------

def coverage_threshold_analysis(
    study: dict | list[list[float]],
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
    confidence: float = 0.95,
) -> list[dict]:
    """Определить, на какой итерации порог покрытия достигается с заданной вероятностью.

    Для порога γ вероятность достижения на итерации k оценивается долей
    экспериментов с I_m(k) ≥ γ; k_дов = первое k, где доля ≥ ``confidence``.
    Если такого k нет (потолок покрытия ниже порога у > 5 % экспериментов),
    k_дов = None — порог недостижим с требуемой вероятностью. Доля достижения
    сопровождается нормальным 95%-м интервалом (Clopper–Pearson не требуется
    при M = 100: np(1−p) ≥ 5 при p ∈ [0.05, 0.95]).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("coverage_threshold_analysis: confidence должно быть в (0, 1).")
    if isinstance(study, dict):
        trajectories = [e["trajectory"] for e in study["experiments"]]
    else:
        trajectories = study
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("coverage_threshold_analysis: ожидаются непустые траектории (M, K).")
    m_total, k_total = arr.shape
    z = 1.96
    results: list[dict] = []
    for gamma in thresholds:
        reached = arr >= gamma
        share = reached.mean(axis=0)
        ks_conf = np.nonzero(share >= confidence)[0]
        first_hits: list[int | None] = []
        for m in range(m_total):
            idx = np.nonzero(reached[m])[0]
            first_hits.append(int(idx[0]) + 1 if idx.size else None)
        finite = [h for h in first_hits if h is not None]
        share_final = float(share[-1])
        half = z * math.sqrt(share_final * (1.0 - share_final) / m_total)
        results.append({
            "threshold": gamma,
            "confidence": confidence,
            "k_confidence": (int(ks_conf[0]) + 1) if ks_conf.size else None,
            "reach_share_final": share_final,
            "reach_share_final_ci95_low": max(0.0, share_final - half),
            "reach_share_final_ci95_high": min(1.0, share_final + half),
            "experiments_reached": len(finite),
            "experiments_total": m_total,
            "k_min_first_reach": min(finite) if finite else None,
            "k_median_first_reach": (float(np.median(finite)) if finite else None),
            "k_max_first_reach": max(finite) if finite else None,
            "share_by_iteration": [float(v) for v in share],
        })
    return results


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ранги значений (средние ранги при совпадениях), без внешних зависимостей."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """ρ Спирмена между двумя рядами (ранговая корреляция, без scipy).

    Константный ряд (нуловая дисперсия рангов) даёт ρ = 0 — ассоциации нет.
    """
    n = len(x)
    if n < 3:
        return math.nan
    rx = _ranks(np.asarray(x, dtype=float))
    ry = _ranks(np.asarray(y, dtype=float))
    if rx[0] == rx[-1] or ry[0] == ry[-1]:
        return 0.0
    d = rx - ry
    return 1.0 - 6.0 * float((d * d).sum()) / (n * (n * n - 1))


def std_trend_analysis(
    iteration_stats: list[dict],
    n_perm: int = 10000,
    seed: int = 20261124,
) -> dict:
    """Проверить по фактическим результатам, убывает ли разброс между экспериментами.

    ρ Спирмена между номером итерации k и std(k); значимость — перестановочный
    тест (n_perm перестановок std по k, фиксированный seed — воспроизводимо).
    Гипотеза «разброс убывает» принимается при ρ < 0 и p < 0.05 — сходимость
    заранее не предполагается: возможен и вывод об отсутствии убывания.
    """
    ks = np.asarray([s["iteration"] for s in iteration_stats], dtype=float)
    stds = np.asarray([s["std"] for s in iteration_stats], dtype=float)
    band = np.asarray([s["p90"] - s["p10"] for s in iteration_stats], dtype=float)
    mask = ~np.isnan(stds)
    ks, stds, band = ks[mask], stds[mask], band[mask]
    if len(ks) < 3:
        raise ValueError("std_trend_analysis: недостаточно точек со значимым std (M ≥ 2).")
    rho = _spearman_rho(ks, stds)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rp = _spearman_rho(ks, stds[rng.permutation(len(stds))])
        if abs(rp) >= abs(rho):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    head = float(stds[:10].mean())
    tail = float(stds[-10:].mean())
    return {
        "n_points": int(len(ks)),
        "spearman_rho_std": rho,
        "permutation_p_value": p_value,
        "decreasing_significant": bool(rho < 0.0 and p_value < 0.05),
        "std_first10_mean": head,
        "std_last10_mean": tail,
        "std_ratio_last_first": (tail / head) if head > 0 else math.nan,
        "band_spearman_rho": _spearman_rho(ks, band),
        "n_perm": n_perm,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Оркестрация исследования
# ---------------------------------------------------------------------------

def run_convergence_study(
    config: ScenarioConfig | dict,
    params: ConvergenceParams,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Выполнить исследование: M независимых экспериментов × K итераций.

    Параллелизм — отдельные процессы (как в stats.run_batch); результаты
    не зависят от числа workers. progress(done, total) вызывается по мере
    завершения экспериментов.
    """
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    if not config_dict.get("frames", {}).get("enabled"):
        raise ConfigError(
            "convergence: исследование сходимости требует включённых кадров РСА "
            "(frames.enabled: true и непустые pass_azimuths_deg)."
        )
    ScenarioConfig.model_validate(config_dict)  # отказ до запуска серии
    if params.iterations < 1:
        raise ValueError("convergence.iterations: число итераций K должно быть не меньше 1.")
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
            experiments.append(run_experiment(config_dict, params, s, experiment_index=i))
            _done()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(run_experiment, config_dict, params, s, experiment_index=i)
                for i, s in enumerate(seeds)
            ]
            for fut in as_completed(futures):
                experiments.append(fut.result())
                _done()

    experiments.sort(key=lambda e: e["experiment"])
    iteration_stats = compute_iteration_stats([e["trajectory"] for e in experiments])
    diagnostics = diagnose_convergence(iteration_stats, params)

    warnings: list[str] = []
    fingerprint = experiments[0]["scenario_fingerprint"]
    for e in experiments[1:]:
        if e["scenario_fingerprint"] != fingerprint:
            warnings.append(
                f"Эксперимент {e['experiment']}: отпечаток сценария отличается от базового — "
                "район/объекты между экспериментами не сохранены."
            )

    return {
        "version": __version__,
        "params": {**asdict(params), "workers": workers},
        "scenario": {
            "name": config_dict.get("name", "district"),
            "seed": config_dict.get("seed"),
            **fingerprint,
        },
        "experiments": experiments,
        "iteration_stats": iteration_stats,
        "convergence": diagnostics,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Экспорт: CSV, JSON, PNG
# ---------------------------------------------------------------------------

def write_convergence_csv(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """Записать CSV: trajectories (строка на эксперимент×итерацию), summary (на итерацию), experiments (на эксперимент)."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    traj_path = out / "convergence_trajectories.csv"
    with traj_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment", "seed", "iteration", "info_completeness",
            "district_coverage", "objects_coverage", "frame_selected", "marginal_gain",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in study["experiments"]:
            for k in range(len(e["trajectory"])):
                writer.writerow({
                    "experiment": e["experiment"],
                    "seed": e["seed"],
                    "iteration": k + 1,
                    "info_completeness": round(e["trajectory"][k], 8),
                    "district_coverage": round(e["district_coverage"][k], 8),
                    "objects_coverage": round(e["objects_coverage"][k], 8),
                    "frame_selected": e["frames_selected"][k],
                    "marginal_gain": round(e["marginal_gains"][k], 8),
                })
    paths["convergence_trajectories.csv"] = traj_path

    summary_path = out / "convergence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "iteration", "mean", "std", "median", "p10", "p90",
            "ci95_low", "ci95_high", "mean_increment",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in study["iteration_stats"]:
            writer.writerow({k: (round(v, 8) if isinstance(v, float) else v) for k, v in s.items()})
    paths["convergence_summary.csv"] = summary_path

    exp_path = out / "convergence_experiments.csv"
    with exp_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["experiment", "seed", "final_completeness", "duration_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in study["experiments"]:
            writer.writerow({
                "experiment": e["experiment"],
                "seed": e["seed"],
                "final_completeness": round(e["final_completeness"], 8),
                "duration_s": round(e["duration_s"], 4),
            })
    paths["convergence_experiments.csv"] = exp_path
    return paths


def write_convergence_json(study: dict, out_dir: str | Path) -> Path:
    """Записать всё исследование (параметры, траектории, статистика, диагноз) в convergence.json."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "convergence.json"
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


_COLOR_REAL = "#4C78A8"
_COLOR_HIGHLIGHT = "#E45756"
_COLOR_ALT = "#F58518"
_COLOR_TARGET = "green"


def render_convergence_charts(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """Построить PNG-графики исследования: среднее с полосой p10–p90, std, все траектории, приросты.

    Полоса разброса — фактические 10–90-процентили по экспериментам на каждой
    итерации; данные не сглаживаются и полоса искусственно не сужается.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    stats = study["iteration_stats"]
    experiments = study["experiments"]
    conv = study["convergence"]
    params = study["params"]
    k_total = len(stats)
    x = np.arange(1, k_total + 1)
    mean = np.asarray([s["mean"] for s in stats], dtype=float)
    median = np.asarray([s["median"] for s in stats], dtype=float)
    p10 = np.asarray([s["p10"] for s in stats], dtype=float)
    p90 = np.asarray([s["p90"] for s in stats], dtype=float)
    ci_lo = np.asarray([s["ci95_low"] for s in stats], dtype=float)
    ci_hi = np.asarray([s["ci95_high"] for s in stats], dtype=float)
    std = np.asarray([s["std"] for s in stats], dtype=float)
    increments = np.asarray([s["mean_increment"] for s in stats], dtype=float)
    k_stab = conv["stabilized_at_iteration"]

    # 1) Средняя полнота vs число итераций: полоса p10–p90 и 95%-й ДИ среднего
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(x, p10, p90, color=_COLOR_REAL, alpha=0.25,
                    label=f"разброс p10–p90 ({params['experiments']} экспериментов)")
    ax.fill_between(x, ci_lo, ci_hi, color=_COLOR_REAL, alpha=0.45,
                    label="95%-й ДИ среднего")
    ax.plot(x, mean, color=_COLOR_REAL, linewidth=2.0, marker="o", markersize=3.5,
            label="средняя полнота I(k)")
    ax.plot(x, median, color=_COLOR_ALT, linestyle="--", linewidth=1.2, label="медиана")
    ax.axhline(params["min_completeness"], color=_COLOR_TARGET, linestyle=":", linewidth=1.4,
               label=f"требуемая полнота {params['min_completeness']:.3g}")
    if k_stab is not None:
        ax.axvline(k_stab, color=_COLOR_HIGHLIGHT, linestyle="--", linewidth=1.2,
                   label=f"стабилизация на k = {k_stab}")
    ax.set_title(f"Средняя полнота I(k) vs число итераций — диагноз: {conv['verdict']}")
    ax.set_xlabel("число итераций k")
    ax.set_ylabel("полнота информации I(k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, 0.02, conv["verdict_ru"] + ".", fontsize=8, alpha=0.75)
    fig.text(0.01, 0.005,
             "Полосы — фактические p10–p90 и 95%-й ДИ среднего по экспериментам на каждой итерации; сглаживание не применяется.",
             fontsize=7, alpha=0.7)
    fig.tight_layout()
    p = out / "convergence_mean_spread.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_mean_spread.png"] = p

    # 2) Стандартное отклонение между экспериментами vs число итераций
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, std, color=_COLOR_REAL, linewidth=2.0, marker="s", markersize=3.5,
            label="std между экспериментами (ddof=1)")
    ax.plot(x, p90 - p10, color=_COLOR_ALT, linestyle="--", linewidth=1.2, label="ширина p90 − p10")
    ax.axhline(params["spread_tol"], color=_COLOR_HIGHLIGHT, linestyle=":", linewidth=1.4,
               label=f"порог разброса {params['spread_tol']:.3g}")
    if k_stab is not None:
        ax.axvline(k_stab, color=_COLOR_HIGHLIGHT, linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"стабилизация на k = {k_stab}")
    ax.set_title("Изменение стандартного отклонения при увеличении числа итераций")
    ax.set_xlabel("число итераций k")
    ax.set_ylabel("разброс полноты между экспериментами")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "convergence_std.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_std.png"] = p

    # 3) Все траектории I(k) — честное отображение нестабильности и локальных решений
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for e in experiments:
        ax.plot(x, e["trajectory"], color=_COLOR_REAL, linewidth=0.8, alpha=0.35)
    ax.plot(x, mean, color=_COLOR_HIGHLIGHT, linewidth=2.0, label="средняя полнота I(k)")
    ax.axhline(params["min_completeness"], color=_COLOR_TARGET, linestyle=":", linewidth=1.4,
               label=f"требуемая полнота {params['min_completeness']:.3g}")
    ax.set_title(f"Траектории полноты по экспериментам (M = {params['experiments']})")
    ax.set_xlabel("число итераций k")
    ax.set_ylabel("полнота информации I(k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out / "convergence_trajectories.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_trajectories.png"] = p

    # 4) Средний прирост полноты vs число итераций (критерий стабилизации)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, increments, color=_COLOR_REAL, alpha=0.85, edgecolor="black", linewidth=0.4,
           label="средний прирост I(k) − I(k−1)")
    ax.axhline(params["stabilization_tol"], color=_COLOR_HIGHLIGHT, linestyle=":", linewidth=1.4,
               label=f"порог стабилизации {params['stabilization_tol']:.3g}")
    if k_stab is not None:
        ax.axvline(k_stab, color=_COLOR_HIGHLIGHT, linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"стабилизация на k = {k_stab}")
    ax.set_title("Средний прирост полноты информации по итерациям")
    ax.set_xlabel("число итераций k")
    ax.set_ylabel("средний прирост")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "convergence_increments.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_increments.png"] = p

    return paths


def render_comparison_charts(studies: list[tuple[str, dict]], out_dir: str | Path) -> dict[str, Path]:
    """Сравнение исследований (метка, study): кривые средней полноты и std, сводные метрики.

    Кривые — фактические средние и std по итерациям каждого исследования,
    сглаживание не применяется.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not studies:
        return paths

    colors = plt.cm.tab10.colors

    # 1) Кривые средней полноты и разброса
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, (label, study) in enumerate(studies):
        stats = study["iteration_stats"]
        x = np.asarray([s["iteration"] for s in stats], dtype=float)
        mean = np.asarray([s["mean"] for s in stats], dtype=float)
        std = np.asarray([s["std"] for s in stats], dtype=float)
        c = colors[i % 10]
        a1.plot(x, mean, color=c, linewidth=1.8, label=label)
        a2.plot(x, std, color=c, linewidth=1.8, label=label)
    min_comp = studies[0][1]["params"]["min_completeness"]
    for ax, ylabel in ((a1, "средняя полнота I(k)"), (a2, "std между экспериментами")):
        ax.set_xlabel("число итераций k")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    a1.axhline(min_comp, color=_COLOR_TARGET, linestyle=":", linewidth=1.2,
               label=f"требуемая полнота {min_comp:.3g}")
    a1.set_title("Средняя полнота информации vs число итераций")
    a2.set_title("Стандартное отклонение между экспериментами")
    fig.tight_layout()
    p = out / "convergence_comparison_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_comparison_curves.png"] = p

    # 2) Сводные метрики: итоговая полнота (±95% ДИ), итоговый std, AUC средней кривой, k стабилизации
    labels = [s[0] for s in studies]
    finals_mean = [s[1]["iteration_stats"][-1]["mean"] for s in studies]
    finals_ci = []
    for _, study in studies:
        s = study["iteration_stats"][-1]
        finals_ci.append((s["ci95_high"] - s["ci95_low"]) / 2.0)
    finals_std = [s[1]["iteration_stats"][-1]["std"] for s in studies]
    aucs = []
    for _, study in studies:
        stats = study["iteration_stats"]
        mean = np.asarray([s["mean"] for s in stats], dtype=float)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 / < 2.0
        aucs.append(float(trapezoid(mean, dx=1.0) / len(mean)))
    k_stabs = [s[1]["convergence"]["stabilized_at_iteration"] or 0 for s in studies]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    xpos = np.arange(len(labels))
    for ax, values, title in (
        (axes[0], finals_mean, "Итоговая средняя полнота I(K)"),
        (axes[1], finals_std, "Итоговый std между экспериментами"),
        (axes[2], aucs, "AUC средней кривой (скорость роста)"),
        (axes[3], k_stabs, "Итерация стабилизации k*"),
    ):
        bars = ax.bar(xpos, values, color=[colors[i % 10] for i in range(len(labels))],
                      edgecolor="black")
        if title.startswith("Итоговая средняя"):
            ax.errorbar(xpos, values, yerr=finals_ci, fmt="none", ecolor="black", capsize=3)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.4g}" if v < 10 else f"{v:.0f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = out / "convergence_comparison_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["convergence_comparison_summary.png"] = p

    return paths
