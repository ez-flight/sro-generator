"""Серия параллельных прогонов генерации и статистика по ним.

Каждый прогон — независимая генерация сценария со своим seed (seed_start + i).
Прогоны выполняются в отдельных процессах; из дочерних процессов
возвращаются только plain-данные (без shapely-геометрий).
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ScenarioConfig
from .generator import generate_scenario
from .placement import QUADRANT_GRID  # единый источник константы сетки

__all__ = [
    "QUADRANT_GRID",
    "run_single", "run_batch", "summarize", "write_batch_csv",
    "chi2_uniform", "chi2_gof", "ks_test", "chi2_sf", "kolmogorov_sf", "clark_evans_index",
    "simulate_reference_quadrant_counts",
    "render_report_charts", "render_analytics_charts",
    "render_series_comparison", "render_benchmark_charts",
]

# Минимальная ожидаемая частота для применимости асимптотического χ² (правило Кохрана)
_CHI2_MIN_EXPECTED = 1.0
_CHI2_WARN_EXPECTED = 5.0
# Минимальный объём выборки для приемлемой точности асимптотического p-value KS
_KS_MIN_N = 20


# ---------------------------------------------------------------------------
# Специальные функции: регуляризованная неполная гамма-функция и
# функция распределения Колмогорова (без внешних зависимостей).
# Алгоритмы: Numerical Recipes (ряд/непрерывная дробь), Марсалья–Тян–Ванга для KS.
# ---------------------------------------------------------------------------

_GAMMA_ITMAX = 1000
_GAMMA_EPS = 3e-14
_GAMMA_FPMIN = 1e-300


def _gammq(a: float, x: float) -> float:
    """Регуляризованная верхняя неполная гамма-функция Q(a, x) = Γ(a,x)/Γ(a)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("gammq: недопустимые аргументы.")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:  # ряд для P, затем Q = 1 − P
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(_GAMMA_ITMAX):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * _GAMMA_EPS:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return min(max(1.0 - p, 0.0), 1.0)
    # непрерывная дробь для Q
    b = x + 1.0 - a
    c = 1.0 / _GAMMA_FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _GAMMA_ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _GAMMA_FPMIN:
            d = _GAMMA_FPMIN
        c = b + an / c
        if abs(c) < _GAMMA_FPMIN:
            c = _GAMMA_FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_EPS:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return min(max(q, 0.0), 1.0)


def chi2_sf(statistic: float, df: int) -> float:
    """Выживание χ²-распределения: P(X ≥ statistic), X ~ χ²(df)."""
    if df < 1:
        return math.nan
    return _gammq(df / 2.0, statistic / 2.0)


def kolmogorov_sf(lam: float, terms: int = 200) -> float:
    """Выживание распределения Колмогорова: Q(λ) = 2 Σ (−1)^(k−1) e^(−2k²λ²)."""
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for k in range(1, terms + 1):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
        total += term
        if abs(term) < 1e-15:
            break
    return min(max(total, 0.0), 1.0)


def chi2_gof(counts: list[int], expected: list[float]) -> dict:
    """Критерий согласия χ² Пирсона против заданных ожидаемых частот.

    Проверка согласия наблюдаемых частот гипотезе, задающей ``expected``
    (не доказательство равномерности). Возвращает статистику, число степеней
    свободы, p-value и предупреждения о применимости асимптотического приближения.
    """
    if len(counts) != len(expected):
        raise ValueError("chi2_gof: длины counts и expected должны совпадать.")
    if any(e < 0 for e in expected):
        raise ValueError("chi2_gof: ожидаемые частоты не могут быть отрицательными.")
    n_obs = sum(counts)
    n_exp = sum(expected)
    warnings: list[str] = []
    rescaled = list(expected)
    if n_exp > 0 and abs(n_obs - n_exp) > 1e-9 * max(1.0, n_obs):
        warnings.append(
            f"суммы наблюдаемых ({n_obs:g}) и ожидаемых ({n_exp:g}) частот различаются; "
            "ожидаемые пропорционально перенормированы."
        )
        rescaled = [e * n_obs / n_exp for e in expected]
    usable = [(int(o), float(e)) for o, e in zip(counts, rescaled) if e > 0]
    dropped = len(expected) - len(usable)
    if dropped:
        warnings.append(f"{dropped} ячеек с нулевой ожидаемой частотой исключены.")
    if not usable:
        return {
            "kind": "chi2", "statistic": math.nan, "df": 0, "p_value": math.nan,
            "n": n_obs, "min_expected": 0.0, "applicable": False,
            "warnings": warnings + ["нет ячеек с положительной ожидаемой частотой."],
            "interpretation": "проверка согласия с заданной гипотезой (не доказательство равномерности)",
        }
    df = len(usable) - 1
    statistic = float(sum((o - e) ** 2 / e for o, e in usable))
    min_expected = min(e for _, e in usable)
    small = sum(1 for _, e in usable if e < _CHI2_WARN_EXPECTED)
    if min_expected < _CHI2_MIN_EXPECTED:
        warnings.append(
            f"минимальная ожидаемая частота {min_expected:.3g} < {_CHI2_MIN_EXPECTED:g}: "
            "асимптотическое χ²-приближение недостоверно, используйте точный тест или симуляцию."
        )
    elif small > 0.2 * len(usable):
        warnings.append(
            f"доля ячеек с ожидаемой частотой < {_CHI2_WARN_EXPECTED:g} составляет "
            f"{small}/{len(usable)}: приближение грубое."
        )
    p_value = chi2_sf(statistic, df) if df >= 1 else math.nan
    if df < 1:
        warnings.append("число степеней свободы < 1, p-value не определён.")
    return {
        "kind": "chi2",
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "n": n_obs,
        "min_expected": min_expected,
        "applicable": df >= 1 and min_expected >= _CHI2_MIN_EXPECTED,
        "warnings": warnings,
        "interpretation": "проверка согласия с заданной гипотезой (не доказательство равномерности)",
    }


def ks_test(sample: list[float], cdf: Callable[[float], float], *, discrete: bool = False) -> dict:
    """Критерий Колмогорова–Смирнова согласия выборки непрерывной F (cdf).

    Не применяйте к дискретным распределениям: при ``discrete=True`` возвращает
    ``applicable=False`` с предупреждением. Асимптотическое p-value (Марсалья–Тян–Ванга)
    грубо при малых выборках — меньше ``_KS_MIN_N`` добавляется предупреждение.
    """
    if discrete:
        return {
            "kind": "ks", "statistic": math.nan, "n": len(sample), "p_value": math.nan,
            "applicable": False,
            "warnings": ["дискретное распределение: обычный непрерывный критерий KS неприменим "
                         "(нужен дискретный вариант или χ²)."],
            "interpretation": "проверка согласия с заданной гипотезой (не доказательство равномерности)",
        }
    xs = sorted(float(v) for v in sample)
    n = len(xs)
    warnings: list[str] = []
    if n < 2:
        return {
            "kind": "ks", "statistic": math.nan, "n": n, "p_value": math.nan,
            "applicable": False, "warnings": warnings + ["выборка меньше 2 элементов."],
            "interpretation": "проверка согласия с заданной гипотезой (не доказательство равномерности)",
        }
    if len(set(xs)) < n:
        warnings.append("в выборке есть совпадающие значения: эффективный объём меньше n.")
    if n < _KS_MIN_N:
        warnings.append(
            f"n = {n} < {_KS_MIN_N}: асимптотическое p-value грубое, выводы делайте осторожно."
        )
    d_plus = 0.0
    d_minus = 0.0
    for i, x in enumerate(xs):
        f = cdf(x)
        d_plus = max(d_plus, (i + 1) / n - f)
        d_minus = max(d_minus, f - i / n)
    d_stat = max(d_plus, d_minus)
    lam = d_stat * (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
    p_value = kolmogorov_sf(lam)
    return {
        "kind": "ks",
        "statistic": d_stat,
        "statistic_plus": d_plus,
        "statistic_minus": d_minus,
        "n": n,
        "p_value": p_value,
        "applicable": True,
        "warnings": warnings,
        "interpretation": "проверка согласия с заданной гипотезой (не доказательство равномерности)",
    }


def clark_evans_index(nnd_mean_m: float, district_area_m2: float, n_objects: int) -> float | None:
    """Индекс Кларка–Эванса R = mean(NND) / E_csr[NND], E_csr = 0.5·√(A/n).

    R ≈ 1 — соответствие однородному пуассоновскому процессу; < 1 — кластеризация;
    > 1 — регулярность. При min_distance_m > 0 предпосылки CSR нарушены —
    интерпретировать только как описательную метрику.
    """
    if n_objects < 2 or district_area_m2 <= 0.0:
        return None
    expected = 0.5 * math.sqrt(district_area_m2 / n_objects)
    if expected <= 0.0:
        return None
    return nnd_mean_m / expected


def _quadrant_counts(scenario, grid: int) -> list[int]:
    """Число центров объектов в каждой ячейке сетки grid x grid над границами района."""
    minx, miny, maxx, maxy = scenario.district_local.bounds
    sx = (maxx - minx) / grid
    sy = (maxy - miny) / grid
    counts = [0] * (grid * grid)
    for o in scenario.objects:
        ix = min(int((o.center.x - minx) / sx), grid - 1)
        iy = min(int((o.center.y - miny) / sy), grid - 1)
        counts[iy * grid + ix] += 1
    return counts


def run_single(config_dict: dict, seed: int) -> dict:
    """Один прогон генерации. Любой сбой фиксируется в результате (ok=False), не поднимается."""
    started = time.perf_counter()
    cfg = dict(config_dict)
    cfg["seed"] = seed
    try:
        config = ScenarioConfig.model_validate(cfg)
        scenario = generate_scenario(config)
    except Exception as exc:
        return {
            "seed": seed,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": time.perf_counter() - started,
        }
    duration = time.perf_counter() - started

    centers = np.array([[o.center.x, o.center.y] for o in scenario.objects])
    nnd: list[float] = []
    if len(centers) > 1:
        dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        nnd = dist.min(axis=1).tolist()

    district_area = float(scenario.district_local.area)
    objects_area = float(sum(o.area_m2 for o in scenario.objects))

    return {
        "seed": seed,
        "ok": True,
        "error": None,
        "duration_s": duration,
        "attempts_total": scenario.report.attempts_total,
        "exact_checks_total": scenario.report.exact_checks_total,
        "tree_queries": scenario.report.tree_queries,
        "tree_rebuilds": scenario.report.tree_rebuilds,
        "center_region_area_mean_m2": scenario.report.center_region_area_mean_m2,
        "grid_cell_area_fractions": scenario.report.grid_cell_area_fractions_mean(),
        "warnings": list(scenario.warnings),
        "district_area_m2": district_area,
        "objects_total_area_m2": objects_area,
        "fill_ratio": objects_area / district_area if district_area > 0 else 0.0,
        "nnd_mean_m": float(np.mean(nnd)) if nnd else 0.0,
        "nnd_min_m": float(np.min(nnd)) if nnd else 0.0,
        "quadrant_counts": _quadrant_counts(scenario, QUADRANT_GRID),
        "coverage": [f.coverage_fraction for f in scenario.frames],
        "full_coverage_frames": sum(1 for f in scenario.frames if f.coverage_fraction >= 0.999),
        "objects": [
            {
                "object_id": o.object_id,
                "area_m2": o.area_m2,
                "vertex_count": o.vertex_count,
                "aspect_ratio": _aspect_ratio(o),
                "class_label": o.class_label,
                "priority": o.priority,
                "deadline_h": o.deadline_h,
                "resolution_m": o.resolution_m,
                "rotation_deg": o.rotation_deg,
                "placement_mode": o.placement_mode,
                "coordinate_fixed": o.coordinate_fixed,
                "cx_m": o.center.x,
                "cy_m": o.center.y,
            }
            for o in scenario.objects
        ],
    }


def _aspect_ratio(o) -> float:
    """Отношение сторон ориентированного описанного прямоугольника объекта."""
    rect = o.geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    if len(coords) < 3:
        return 1.0
    sides = [math.hypot(coords[(i + 1) % len(coords)][0] - coords[i][0],
                        coords[(i + 1) % len(coords)][1] - coords[i][1])
             for i in range(len(coords))]
    return max(sides) / max(min(sides), 1e-12)


def run_batch(
    config: ScenarioConfig | dict,
    n_runs: int,
    workers: int | None = None,
    seed_start: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    reference_quadrant_counts: list[float] | None = None,
) -> dict:
    """Выполнить n_runs прогонов с seed_start, seed_start+1, ... Параллелизм — отдельные процессы.

    progress(done, total) вызывается по мере завершения прогонов.
    reference_quadrant_counts — эталонное моделируемое распределение по ячейкам 3×3
    (см. simulate_reference_quadrant_counts) для χ²-проверки пространственной
    равномерности вместо площадной взвески.
    Возвращает {"runs": [...], "summary": {...}}; упавшие прогоны не роняют серию.
    """
    config_dict = config.to_dict() if isinstance(config, ScenarioConfig) else dict(config)
    if seed_start is None:
        seed_start = int(config_dict.get("seed", 0))
    if workers is None:
        workers = min(max(1, n_runs), os.cpu_count() or 1)
    workers = max(1, min(workers, n_runs))

    seeds = [seed_start + i for i in range(n_runs)]
    runs: list[dict] = []

    def _done() -> None:
        if progress:
            progress(len(runs), n_runs)

    if workers == 1:
        for s in seeds:
            runs.append(run_single(config_dict, s))
            _done()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single, config_dict, s) for s in seeds]
            for fut in as_completed(futures):
                runs.append(fut.result())
                _done()

    runs.sort(key=lambda r: r["seed"])
    return {"runs": runs, "summary": summarize(runs, config_dict, reference_quadrant_counts=reference_quadrant_counts)}


def chi2_uniform(counts: list[int], expected_per_bin: float) -> float:
    """Статистика хи-квадрат равномерности; ~1.0 на степень свободы означает согласие."""
    if expected_per_bin <= 0:
        return 0.0
    return float(sum((c - expected_per_bin) ** 2 / expected_per_bin for c in counts))


def simulate_reference_quadrant_counts(
    config_dict: dict,
    n_sim: int = 30,
    workers: int = 1,
    seed_start: int | None = None,
) -> list[float]:
    """Моделируемое эталонное среднее распределение центров по ячейкам 3×3.

    Генерирует серию прогонов с той же конфигурацией и возвращает средние
    ожидаемые частоты по ячейкам. Используется как эталонная гипотеза для
    χ²-проверки пространственной равномерности вместо наивной равномерной
    модели, когда ограничения размещения (min_distance_m, размеры объектов)
    принципиально меняют распределение принимаемых центров.
    """
    sim = run_batch(config_dict, n_runs=n_sim, workers=workers, seed_start=seed_start)
    ok = [r for r in sim["runs"] if r["ok"]]
    if not ok:
        return []
    cells = QUADRANT_GRID * QUADRANT_GRID
    return [float(np.mean([r["quadrant_counts"][i] for r in ok])) for i in range(cells)]


def summarize(runs: list[dict], config_dict: dict, reference_quadrant_counts: list[float] | None = None) -> dict:
    """Сводные показатели серии: успешность, попытки, плотность, покрытие, равномерность.

    Статистические проверки — это проверки согласия с заявленной гипотезой,
    а не доказательства равномерности: ограничения размещения (min_distance_m,
    размеры объектов, отступы) систематически меняют распределение принимаемых
    объектов, что отражается в предупреждениях и флаге применимости.
    """
    ok = [r for r in runs if r["ok"]]
    failed = [r for r in runs if not r["ok"]]

    def _stat(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    objs_cfg = config_dict.get("objects", {})
    attrs_cfg = config_dict.get("attributes", {})
    placement_cfg = config_dict.get("placement", {})
    grid_cells = QUADRANT_GRID * QUADRANT_GRID

    vertex_counts: Counter = Counter()
    class_counts: Counter = Counter()
    priority_counts: Counter = Counter()
    deadline_counts: Counter = Counter()
    resolution_counts: Counter = Counter()
    for r in ok:
        for o in r["objects"]:
            vertex_counts[o["vertex_count"]] += 1
            class_counts[o["class_label"]] += 1
            priority_counts[o["priority"]] += 1
            deadline_counts[o["deadline_h"]] += 1
            resolution_counts[o["resolution_m"]] += 1
    all_objects = [o for r in ok for o in r["objects"]]
    n_objects = len(all_objects)

    n_vertices_options = objs_cfg.get("vertices_max", 10) - objs_cfg.get("vertices_min", 3) + 1
    labels = list(attrs_cfg.get("class_labels", []))
    priorities = list(range(attrs_cfg.get("priority_min", 1), attrs_cfg.get("priority_max", 5) + 1))
    deadlines = list(attrs_cfg.get("deadline_choices_h", []))
    resolutions = list(attrs_cfg.get("resolution_choices_m", []))

    coverage = [c for r in ok for c in r["coverage"]]
    frames_total = sum(len(r["coverage"]) for r in ok)

    statistical_tests: dict[str, dict] = {}

    # --- χ² по дискретным равномерным распределениям ---
    if n_vertices_options > 1 and n_objects:
        v_options = list(range(objs_cfg.get("vertices_min", 3), objs_cfg.get("vertices_max", 10) + 1))
        statistical_tests["vertices"] = chi2_gof(
            [vertex_counts.get(v, 0) for v in v_options],
            [n_objects / len(v_options)] * len(v_options),
        )
    else:
        statistical_tests["vertices"] = _not_applicable("chi2", n_objects, "одна опция вершин — проверка не требуется.")

    if labels:
        statistical_tests["classes"] = chi2_gof(
            [class_counts.get(x, 0) for x in labels],
            [n_objects / len(labels)] * len(labels),
        )
    else:
        statistical_tests["classes"] = _not_applicable("chi2", n_objects, "категории не заданы.")

    if len(priorities) > 1 and n_objects:
        statistical_tests["priority"] = chi2_gof(
            [priority_counts.get(p, 0) for p in priorities],
            [n_objects / len(priorities)] * len(priorities),
        )
    else:
        statistical_tests["priority"] = _not_applicable("chi2", n_objects, "одно значение приоритета — проверка не требуется.")

    if deadlines and n_objects:
        statistical_tests["deadline"] = chi2_gof(
            [deadline_counts.get(float(d), 0) for d in deadlines],
            [n_objects / len(deadlines)] * len(deadlines),
        )
    else:
        statistical_tests["deadline"] = _not_applicable("chi2", n_objects, "варианты сроков не заданы.")

    if resolutions and n_objects:
        statistical_tests["resolution"] = chi2_gof(
            [resolution_counts.get(float(r), 0) for r in resolutions],
            [n_objects / len(resolutions)] * len(resolutions),
        )
    else:
        statistical_tests["resolution"] = _not_applicable("chi2", n_objects, "варианты разрешений не заданы.")

    # --- KS по непрерывным равномерным распределениям ---
    area_min = float(objs_cfg.get("area_min_m2", 0.0))
    area_max = float(objs_cfg.get("area_max_m2", 0.0))
    if area_max > area_min and n_objects:
        areas = [o["area_m2"] for o in all_objects]
        t = ks_test(areas, lambda x: min(max((x - area_min) / (area_max - area_min), 0.0), 1.0))
        t["warnings"].append(
            "выборка успешно размещённых объектов: отбраковка при размещении зависит от площади, "
            "поэтому согласие с исходным U[area_min, area_max] может нарушаться без ошибки генератора."
        )
        statistical_tests["areas"] = t
    else:
        statistical_tests["areas"] = _not_applicable("ks", n_objects, "выражденный диапазон площадей.")

    rot_min = float(objs_cfg.get("rotation_min_deg", 0.0))
    rot_max = float(objs_cfg.get("rotation_max_deg", 360.0))
    if rot_max > rot_min and n_objects:
        rotations = [o["rotation_deg"] for o in all_objects]
        t = ks_test(rotations, lambda x: min(max((x - rot_min) / (rot_max - rot_min), 0.0), 1.0))
        t["warnings"].append(
            "поворот влияет на площадь допустимой области центров: принятые значения слегка смещены "
            "к ориентациям с большей областью размещения."
        )
        statistical_tests["rotations"] = t
    else:
        statistical_tests["rotations"] = _not_applicable("ks", n_objects, "совпадающие пределы поворота — выборка дискретна/вырождена.")

    # --- пространственная равномерность по сетке 3×3 ---
    # Ожидаемые частоты: либо эталонное моделируемое распределение, либо
    # площадная взвеска по средним долям допустимой области в ячейках.
    quadrant_results = []
    used_area_weighting = False
    for r in ok:
        n_obj = len(r["objects"])
        if n_obj == 0:
            continue
        if reference_quadrant_counts and len(reference_quadrant_counts) == grid_cells:
            expected = [c * n_obj / max(1e-12, sum(reference_quadrant_counts)) for c in reference_quadrant_counts]
            res = chi2_gof(list(r["quadrant_counts"]), expected)
            res["warnings"].append("ожидаемые частоты — эталонное моделируемое распределение (симуляция).")
            quadrant_results.append(res)
        else:
            fr = r.get("grid_cell_area_fractions") or []
            if len(fr) == grid_cells and sum(fr) > 0:
                used_area_weighting = True
                res = chi2_gof(list(r["quadrant_counts"]), [f * n_obj for f in fr])
                res["warnings"].append(
                    "ожидаемые частоты взвешены площадью допустимой области; min_distance_m и размеры "
                    "объектов дополнительно меняют распределение — интерпретировать как первый порядок."
                )
                quadrant_results.append(res)
            else:
                res = chi2_gof(list(r["quadrant_counts"]), [n_obj / grid_cells] * grid_cells)
                res["warnings"].append("равномерная модель по ограничивающему прямоугольнику — грубое приближение.")
                quadrant_results.append(res)
    if quadrant_results:
        statistical_tests["quadrant_grid"] = {
            "kind": "chi2",
            "statistic": float(np.mean([q["statistic"] for q in quadrant_results])),
            "df": quadrant_results[0]["df"],
            "p_value": float(np.mean([q["p_value"] for q in quadrant_results])),
            "n": sum(q["n"] for q in quadrant_results),
            "min_expected": float(np.mean([q["min_expected"] for q in quadrant_results])),
            "applicable": all(q["applicable"] for q in quadrant_results),
            "warnings": sorted({w for q in quadrant_results for w in q["warnings"]}),
            "interpretation": "проверка согласия с гипотезой пространственного распределения центров",
        }
    else:
        statistical_tests["quadrant_grid"] = _not_applicable("chi2", 0, "нет успешных прогонов с объектами.")

    # --- NND и индекс Кларка–Эванса ---
    ce_values = [
        clark_evans_index(r["nnd_mean_m"], r["district_area_m2"], len(r["objects"]))
        for r in ok if len(r["objects"]) > 1
    ]
    ce_values = [v for v in ce_values if v is not None]
    min_distance_m = float(placement_cfg.get("min_distance_m", 0.0))
    ce_warnings = []
    if min_distance_m > 0:
        ce_warnings.append(
            "min_distance_m > 0 нарушает предпосылку однородного пуассоновского процесса: "
            "индекс — описательная метрика, а не строгая проверка CSR."
        )

    def _chi2_df(test: dict) -> float:
        if not test.get("applicable") or not test.get("df"):
            return 0.0
        return float(test["statistic"]) / max(1, test["df"])

    return {
        "n_runs": len(runs),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "errors": [{"seed": r["seed"], "error": r["error"]} for r in failed],
        "attempts_total": _stat([r["attempts_total"] for r in ok]),
        "exact_checks_total": _stat([r.get("exact_checks_total", 0) for r in ok]),
        "duration_s": _stat([r["duration_s"] for r in ok]),
        "fill_ratio": _stat([r["fill_ratio"] for r in ok]),
        "nnd_mean_m": _stat([r["nnd_mean_m"] for r in ok]),
        "nnd_min_m": _stat([r["nnd_min_m"] for r in ok]),
        "clark_evans_r": _stat(ce_values),
        "clark_evans_warnings": ce_warnings,
        "coverage_fraction": _stat(coverage),
        "full_coverage_share": (sum(r["full_coverage_frames"] for r in ok) / frames_total) if frames_total else 0.0,
        "areas_m2": _stat([o["area_m2"] for o in all_objects]),
        "aspect_ratio": _stat([o.get("aspect_ratio", 0.0) for o in all_objects]),
        "warnings_total": sum(len(r["warnings"]) for r in ok),
        "vertex_counts": dict(sorted(vertex_counts.items())),
        "vertex_chi2_df": _chi2_df(statistical_tests["vertices"]),
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: str(kv[0]))),
        "class_chi2_df": _chi2_df(statistical_tests["classes"]),
        "quadrant_chi2_df": _chi2_df(statistical_tests["quadrant_grid"]),
        "quadrant_area_weighted": used_area_weighting or bool(reference_quadrant_counts),
        "statistical_tests": statistical_tests,
        "config": {
            "min_distance_m": placement_cfg.get("min_distance_m", 0.0),
            "area_min_m2": objs_cfg.get("area_min_m2", 0.0),
            "area_max_m2": objs_cfg.get("area_max_m2", 0.0),
            "vertices_min": objs_cfg.get("vertices_min", 3),
            "vertices_max": objs_cfg.get("vertices_max", 10),
            "class_labels": labels,
            "count": objs_cfg.get("count", 0),
            "max_aspect_ratio": objs_cfg.get("max_aspect_ratio", 0.0),
            "priority_min": attrs_cfg.get("priority_min", 1),
            "priority_max": attrs_cfg.get("priority_max", 5),
            "deadline_choices_h": [float(d) for d in deadlines],
            "resolution_choices_m": [float(r) for r in resolutions],
            "rotation_min_deg": objs_cfg.get("rotation_min_deg", 0.0),
            "rotation_max_deg": objs_cfg.get("rotation_max_deg", 360.0),
            "center_region_mode": placement_cfg.get("center_region_mode", "exact"),
            "spatial_index": placement_cfg.get("spatial_index", True),
        },
    }


def _not_applicable(kind: str, n: int, reason: str) -> dict:
    """Заглушка теста, который неприменим в данной конфигурации."""
    return {
        "kind": kind,
        "statistic": math.nan,
        "df": 0,
        "p_value": math.nan,
        "n": n,
        "min_expected": 0.0,
        "applicable": False,
        "warnings": [reason],
        "interpretation": "проверка не выполнялась",
    }


def write_batch_csv(batch: dict, out_dir: str | Path) -> dict[str, Path]:
    """Записать CSV для внешнего анализа: runs_summary.csv (строка на прогон) и objects_all.csv (строка на объект)."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = batch["runs"]
    paths: dict[str, Path] = {}

    summary_path = out / "runs_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "seed", "ok", "error", "duration_s", "attempts_total",
            "objects_total_area_m2", "district_area_m2", "fill_ratio",
            "nnd_mean_m", "nnd_min_m", "n_objects", "frames_total",
            "full_coverage_frames", "warnings_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({
                "seed": r["seed"],
                "ok": r["ok"],
                "error": r.get("error") or "",
                "duration_s": round(r["duration_s"], 4),
                "attempts_total": r.get("attempts_total", ""),
                "objects_total_area_m2": round(r.get("objects_total_area_m2", 0.0), 1),
                "district_area_m2": round(r.get("district_area_m2", 0.0), 1),
                "fill_ratio": round(r.get("fill_ratio", 0.0), 6),
                "nnd_mean_m": round(r.get("nnd_mean_m", 0.0), 2),
                "nnd_min_m": round(r.get("nnd_min_m", 0.0), 2),
                "n_objects": len(r.get("objects", [])),
                "frames_total": len(r.get("coverage", [])),
                "full_coverage_frames": r.get("full_coverage_frames", ""),
                "warnings_count": len(r.get("warnings", [])),
            })
    paths["runs_summary.csv"] = summary_path

    objects_path = out / "objects_all.csv"
    with objects_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "seed", "object_id", "class_label", "area_m2", "vertex_count",
            "aspect_ratio", "priority", "deadline_h", "resolution_m", "rotation_deg",
            "placement_mode", "coordinate_fixed", "cx_m", "cy_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            if not r["ok"]:
                continue
            for o in r["objects"]:
                row = {k: o[k] for k in fields if k != "seed"}
                row["seed"] = r["seed"]
                writer.writerow(row)
    paths["objects_all.csv"] = objects_path
    return paths


def render_report_charts(batch: dict, out_dir: str | Path) -> dict[str, Path]:
    """Построить PNG-графики по серии: размещение, атрибуты, геометрия, кадры/пространство."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in batch["runs"] if r["ok"]]
    s = batch["summary"]
    cfg = s["config"]
    paths: dict[str, Path] = {}

    def _hist_bins(arr, max_bins: int) -> int:
        arr = np.asarray(arr, dtype=float)
        span = float(np.ptp(arr))
        scale = max(1.0, float(np.max(np.abs(arr))))
        if span <= 1e-12 * scale:  # вырожденный диапазон (шум float)
            return 1
        return min(max_bins, max(5, len(np.unique(arr)) - 1))

    def _hist(ax, values, title, xlabel, extra_line=None, extra_label=""):
        arr = np.asarray(values, dtype=float)
        ax.hist(arr, bins=_hist_bins(arr, 30), color="#4C78A8", alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.axvline(float(np.mean(values)), color="red", linestyle="--", linewidth=1.2, label=f"среднее {np.mean(values):.3g}")
        if extra_line is not None:
            ax.axvline(extra_line, color="green", linestyle=":", linewidth=1.4, label=extra_label or f"порог {extra_line:.3g}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("прогонов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 1) Размещение: попытки и плотность заполнения
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    if runs:
        _hist(a1, [r["attempts_total"] for r in runs], "Суммарные попытки размещения", "попыток")
        _hist(a2, [r["fill_ratio"] for r in runs], "Доля площади района под объектами", "fill_ratio")
    fig.tight_layout()
    p = out / "chart_placement.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["chart_placement.png"] = p

    # 2) Атрибуты: число вершин и категории против равномерного ожидания
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    v_actual = s["vertex_counts"]
    v_options = list(range(cfg["vertices_min"], cfg["vertices_max"] + 1))
    v_expected = sum(v_actual.values()) / len(v_options)
    a1.bar([str(v) for v in v_options], [v_actual.get(v, 0) for v in v_options], color="#4C78A8", edgecolor="black", label="факт")
    a1.axhline(v_expected, color="green", linestyle="--", label=f"равномерно: {v_expected:.0f}")
    a1.set_title(f"Число вершин (χ²/df = {s['vertex_chi2_df']:.3g})")
    a1.set_xlabel("вершин")
    a1.set_ylabel("объектов")
    a1.grid(True, alpha=0.3)
    a1.legend(fontsize=8)
    c_actual = s["class_counts"]
    labels = cfg["class_labels"]
    c_expected = sum(c_actual.values()) / len(labels) if labels else 0
    a2.bar([str(x) for x in labels], [c_actual.get(x, 0) for x in labels], color="#F58518", edgecolor="black", label="факт")
    if labels:
        a2.axhline(c_expected, color="green", linestyle="--", label=f"равномерно: {c_expected:.0f}")
    a2.set_title(f"Категории объектов (χ²/df = {s['class_chi2_df']:.3g})")
    a2.set_ylabel("объектов")
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = out / "chart_attributes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["chart_attributes.png"] = p

    # 3) Геометрия: площади и расстояния до ближайшего соседа
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    areas = [o["area_m2"] for r in runs for o in r["objects"]]
    if areas:
        a1.hist(np.asarray(areas, dtype=float), bins=_hist_bins(areas, 30), color="#54A24B", alpha=0.85, edgecolor="black", linewidth=0.4)
        a1.axvline(cfg["area_min_m2"], color="red", linestyle=":", linewidth=1.4, label="area_min_m2")
        a1.axvline(cfg["area_max_m2"], color="red", linestyle=":", linewidth=1.4, label="area_max_m2")
        a1.set_title("Площади объектов (все прогоны)")
        a1.set_xlabel("м²")
        a1.set_ylabel("объектов")
        a1.grid(True, alpha=0.3)
        a1.legend(fontsize=8)
        _hist(a2, [r["nnd_mean_m"] for r in runs], "Среднее расстояние до ближайшего соседа", "м",
              extra_line=cfg["min_distance_m"], extra_label="min_distance_m")
    fig.tight_layout()
    p = out / "chart_geometry.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["chart_geometry.png"] = p

    # 4) Кадры и пространственная равномерность
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    coverage = [c for r in runs for c in r["coverage"]]
    if coverage:
        cov_arr = np.asarray(coverage, dtype=float)
        a1.hist(cov_arr, bins=_hist_bins(cov_arr, 25), color="#B279A2", alpha=0.85, edgecolor="black", linewidth=0.4)
        a1.axvline(1.0, color="red", linestyle=":", linewidth=1.4, label="полное покрытие")
        a1.set_title(f"Покрытие объектов кадром (доля полных: {s['full_coverage_share']:.1%})")
        a1.set_xlabel("coverage_fraction")
        a1.set_ylabel("кадров")
        a1.grid(True, alpha=0.3)
        a1.legend(fontsize=8)
    grid_cells = QUADRANT_GRID * QUADRANT_GRID
    cells = [f"({i % QUADRANT_GRID + 1},{i // QUADRANT_GRID + 1})" for i in range(grid_cells)]
    mean_cells = [
        float(np.mean([r["quadrant_counts"][i] for r in runs if r["quadrant_counts"]])) if runs else 0.0
        for i in range(grid_cells)
    ]
    expected_cell = cfg["count"] / grid_cells
    a2.bar(cells, mean_cells, color="#72B7B2", edgecolor="black", label="факт (среднее)")
    a2.axhline(expected_cell, color="green", linestyle="--", label=f"равномерно: {expected_cell:.1f}")
    a2.set_title(f"Объекты по ячейкам района (χ²/df = {s['quadrant_chi2_df']:.3g})")
    a2.set_ylabel("объектов (среднее за прогон)")
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8)
    a2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p = out / "chart_frames_space.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["chart_frames_space.png"] = p

    return paths


# ---------------------------------------------------------------------------
# Аналитические графики (версия 1.1.0). Сохраняются в подкаталог analytics/
# рядом с четырьмя исходными PNG. Цветовое кодирование: стальным — реальные
# измерения, зелёным — теоретические ожидания, оранжевым — результаты
# моделирования/эталонные распределения.
# ---------------------------------------------------------------------------

_COLOR_REAL = "#4C78A8"
_COLOR_THEORY = "green"
_COLOR_SIM = "#F58518"


def _test_annotation(test: dict) -> str:
    """Строка-аннотация результата статистического теста для графика."""
    if not test.get("applicable"):
        return "тест неприменим"
    p = test.get("p_value")
    p_str = f"p = {p:.4g}" if isinstance(p, float) and not math.isnan(p) else "p = n/a"
    stat = test.get("statistic")
    stat_str = f"{stat:.4g}" if isinstance(stat, float) and not math.isnan(stat) else "n/a"
    if test.get("kind") == "chi2":
        return f"χ² = {stat_str}, df = {test.get('df')}, {p_str}"
    return f"D = {stat_str}, {p_str}"


def render_analytics_charts(batch: dict, out_dir: str | Path) -> dict[str, Path]:
    """Построить аналитические PNG-графики по серии прогонов в подкаталог analytics/.

    Реальные измерения отличаются от теоретических ожиданий и результатов
    моделирования цветом и подписями легенды.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "analytics"
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in batch["runs"] if r["ok"]]
    s = batch["summary"]
    cfg = s["config"]
    tests = s.get("statistical_tests", {})
    paths: dict[str, Path] = {}

    def _hist_bins(arr, max_bins: int) -> int:
        arr = np.asarray(arr, dtype=float)
        span = float(np.ptp(arr))
        scale = max(1.0, float(np.max(np.abs(arr))))
        if span <= 1e-12 * scale:
            return 1
        return min(max_bins, max(5, len(np.unique(arr)) - 1))

    all_objects = [o for r in runs for o in r["objects"]]

    # 1) Распределение площадей: реальные vs теоретическая плотность U[min, max]
    fig, ax = plt.subplots(figsize=(8, 5))
    if all_objects:
        areas = np.asarray([o["area_m2"] for o in all_objects], dtype=float)
        bins = _hist_bins(areas, 30)
        counts, edges, _ = ax.hist(areas, bins=bins, color=_COLOR_REAL, alpha=0.85,
                                   edgecolor="black", linewidth=0.4, label="факт (размещённые объекты)")
        a_min, a_max = cfg["area_min_m2"], cfg["area_max_m2"]
        if a_max > a_min:
            pdf = 1.0 / (a_max - a_min)
            xs = np.linspace(a_min, a_max, 200)
            theory_level = float(len(areas) * (edges[1] - edges[0]) * pdf)
            ax.plot(xs, [theory_level] * len(xs), color=_COLOR_THEORY,
                    linestyle="--", linewidth=1.6, label=f"теория: U[{a_min:.4g}, {a_max:.4g}]")
        ax.set_title(f"Распределение площадей объектов ({_test_annotation(tests.get('areas', {}))})")
        ax.set_xlabel("площадь, м²")
        ax.set_ylabel("объектов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_area_distribution.png"] = out / "analytics_area_distribution.png"
    fig.savefig(paths["analytics_area_distribution.png"], dpi=150)
    plt.close(fig)

    # 2) Распределение числа вершин: факт vs равномерное ожидание
    fig, ax = plt.subplots(figsize=(8, 5))
    v_options = list(range(cfg["vertices_min"], cfg["vertices_max"] + 1))
    v_actual = s["vertex_counts"]
    if v_options and all_objects:
        expected = sum(v_actual.values()) / len(v_options)
        ax.bar([str(v) for v in v_options], [v_actual.get(v, 0) for v in v_options],
               color=_COLOR_REAL, edgecolor="black", label="факт")
        ax.axhline(expected, color=_COLOR_THEORY, linestyle="--",
                   label=f"теория: равномерно {expected:.1f}")
        ax.set_title(f"Распределение числа вершин ({_test_annotation(tests.get('vertices', {}))})")
        ax.set_xlabel("число вершин")
        ax.set_ylabel("объектов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_vertex_distribution.png"] = out / "analytics_vertex_distribution.png"
    fig.savefig(paths["analytics_vertex_distribution.png"], dpi=150)
    plt.close(fig)

    # 3) Распределение отношения сторон: факт vs конфигурационный предел
    fig, ax = plt.subplots(figsize=(8, 5))
    if all_objects:
        ratios = np.asarray([o.get("aspect_ratio", 0.0) for o in all_objects], dtype=float)
        ax.hist(ratios, bins=_hist_bins(ratios, 30), color=_COLOR_REAL, alpha=0.85,
                edgecolor="black", linewidth=0.4, label="факт")
        if cfg.get("max_aspect_ratio"):
            ax.axvline(cfg["max_aspect_ratio"], color=_COLOR_THEORY, linestyle="--", linewidth=1.6,
                       label=f"предел max_aspect_ratio = {cfg['max_aspect_ratio']:.3g}")
        ax.set_title("Распределение отношения сторон объектов")
        ax.set_xlabel("отношение сторон (oriented bounding box)")
        ax.set_ylabel("объектов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_aspect_ratio.png"] = out / "analytics_aspect_ratio.png"
    fig.savefig(paths["analytics_aspect_ratio.png"], dpi=150)
    plt.close(fig)

    # 4) Распределение углов поворота: факт vs теоретическая плотность U[min, max]
    fig, ax = plt.subplots(figsize=(8, 5))
    if all_objects:
        rots = np.asarray([o["rotation_deg"] for o in all_objects], dtype=float)
        bins = _hist_bins(rots, 36)
        counts, edges, _ = ax.hist(rots, bins=bins, color=_COLOR_REAL, alpha=0.85,
                                   edgecolor="black", linewidth=0.4, label="факт (размещённые объекты)")
        r_min, r_max = cfg.get("rotation_min_deg", 0.0), cfg.get("rotation_max_deg", 360.0)
        if r_max > r_min:
            xs = np.linspace(r_min, r_max, 200)
            theory_level = float(len(rots) * (edges[1] - edges[0]) / (r_max - r_min))
            ax.plot(xs, [theory_level] * len(xs), color=_COLOR_THEORY,
                    linestyle="--", linewidth=1.6, label=f"теория: U[{r_min:.4g}, {r_max:.4g}]")
        ax.set_title(f"Распределение углов поворота ({_test_annotation(tests.get('rotations', {}))})")
        ax.set_xlabel("угол поворота, град")
        ax.set_ylabel("объектов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_rotation.png"] = out / "analytics_rotation.png"
    fig.savefig(paths["analytics_rotation.png"], dpi=150)
    plt.close(fig)

    # 5) Расстояния до ближайшего соседа (по центрам) против min_distance_m
    fig, ax = plt.subplots(figsize=(8, 5))
    if runs:
        nnd_vals = []
        for r in runs:
            centers = np.array([[o["cx_m"], o["cy_m"]] for o in r["objects"]])
            if len(centers) > 1:
                dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
                np.fill_diagonal(dist, np.inf)
                nnd_vals.extend(dist.min(axis=1).tolist())
        if nnd_vals:
            nnd_arr = np.asarray(nnd_vals, dtype=float)
            ax.hist(nnd_arr, bins=_hist_bins(nnd_arr, 30), color=_COLOR_REAL, alpha=0.85,
                    edgecolor="black", linewidth=0.4, label="факт (NND по центрам)")
            if cfg.get("min_distance_m", 0.0) > 0:
                ax.axvline(cfg["min_distance_m"], color=_COLOR_THEORY, linestyle="--", linewidth=1.6,
                           label=f"теория: минимум min_distance_m = {cfg['min_distance_m']:.4g} м")
            ax.set_title("Распределение расстояний до ближайшего соседа")
            ax.set_xlabel("расстояние, м")
            ax.set_ylabel("объектов")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_nnd.png"] = out / "analytics_nnd.png"
    fig.savefig(paths["analytics_nnd.png"], dpi=150)
    plt.close(fig)

    # 6) Пространственное распределение по сетке 3×3: факт vs площадная взвеска
    #    допустимой области vs наивная равномерная модель
    fig, ax = plt.subplots(figsize=(9, 5))
    grid_cells = QUADRANT_GRID * QUADRANT_GRID
    cells = [f"({i % QUADRANT_GRID + 1},{i // QUADRANT_GRID + 1})" for i in range(grid_cells)]
    if runs:
        pooled = [sum(r["quadrant_counts"][i] for r in runs) for i in range(grid_cells)]
        fr = [r.get("grid_cell_area_fractions") or [] for r in runs]
        fr = [f for f in fr if len(f) == grid_cells and sum(f) > 0]
        x = np.arange(grid_cells)
        ax.bar(x - 0.25, pooled, width=0.25, color=_COLOR_REAL, edgecolor="black",
               label="факт (сумма по прогонам)")
        if fr:
            mean_fr = np.mean(np.asarray(fr, dtype=float), axis=0)
            scaled = mean_fr * sum(pooled)
            ax.bar(x, scaled, width=0.25, color=_COLOR_SIM, edgecolor="black",
                   label="модель: доли площади допустимой области")
        uniform = sum(pooled) / grid_cells
        ax.bar(x + 0.25, [uniform] * grid_cells, width=0.25, color=_COLOR_THEORY, alpha=0.6,
               edgecolor="black", label="теория: равномерно по ограничивающему прямоугольнику")
        ax.set_xticks(x)
        ax.set_xticklabels(cells, rotation=45)
        title = f"Центры по ячейкам 3×3 ({_test_annotation(tests.get('quadrant_grid', {}))})"
        ax.set_title(title)
        ax.set_ylabel("центров объектов")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_grid33.png"] = out / "analytics_grid33.png"
    fig.savefig(paths["analytics_grid33.png"], dpi=150)
    plt.close(fig)

    # 7) Тепловая карта расположения центров объектов
    fig, ax = plt.subplots(figsize=(8, 6))
    if all_objects:
        xs = np.asarray([o["cx_m"] for o in all_objects], dtype=float)
        ys = np.asarray([o["cy_m"] for o in all_objects], dtype=float)
        span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
        heat, xedges, yedges = np.histogram2d(xs, ys, bins=40,
                                              range=[[xs.min() - 0.02 * span, xs.max() + 0.02 * span],
                                                     [ys.min() - 0.02 * span, ys.max() + 0.02 * span]])
        mesh = ax.pcolormesh(xedges, yedges, heat.T, cmap="inferno", shading="auto")
        fig.colorbar(mesh, ax=ax, label="число центров в ячейке")
        ax.set_title(f"Тепловая карта расположения центров ({len(all_objects)} объектов)")
        ax.set_xlabel("X, м (восток)")
        ax.set_ylabel("Y, м (север)")
        ax.set_aspect("equal")
    fig.tight_layout()
    paths["analytics_heatmap.png"] = out / "analytics_heatmap.png"
    fig.savefig(paths["analytics_heatmap.png"], dpi=150)
    plt.close(fig)

    # 8) Попытки размещения в зависимости от заполнения района
    fig, ax = plt.subplots(figsize=(8, 5))
    if runs:
        att = np.asarray([r["attempts_total"] for r in runs], dtype=float)
        fill = np.asarray([r["fill_ratio"] for r in runs], dtype=float)
        ax.scatter(fill, att, color=_COLOR_REAL, s=36, alpha=0.85, edgecolor="black",
                   linewidth=0.4, label="прогоны (реальные измерения)")
        if len(runs) >= 2 and np.ptp(fill) > 0:
            k, b = np.polyfit(fill, att, 1)
            xs = np.linspace(fill.min(), fill.max(), 100)
            ax.plot(xs, k * xs + b, color=_COLOR_SIM, linestyle="--", linewidth=1.4,
                    label=f"линейный тренд: {k:.3g}·fill + {b:.3g}")
        ax.set_title("Попытки размещения в зависимости от заполнения района")
        ax.set_xlabel("fill_ratio (доля площади района под объектами)")
        ax.set_ylabel("суммарные попытки размещения")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_attempts_vs_fill.png"] = out / "analytics_attempts_vs_fill.png"
    fig.savefig(paths["analytics_attempts_vs_fill.png"], dpi=150)
    plt.close(fig)

    # 10) Доля успешных и неуспешных прогонов
    fig, ax = plt.subplots(figsize=(7, 5))
    n_ok, n_failed = s["n_ok"], s["n_failed"]
    bars = ax.bar(["успешные", "неуспешные"], [n_ok, n_failed],
                  color=[_COLOR_REAL, "#E45756"], edgecolor="black")
    for bar, v in zip(bars, [n_ok, n_failed]):
        ax.annotate(f"{v} ({(v / max(1, s['n_runs'])):.1%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Результат серии прогонов: {s['n_runs']} всего")
    ax.set_ylabel("прогонов")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(n_ok, n_failed) * 1.2 + 1)
    fig.tight_layout()
    paths["analytics_success_rate.png"] = out / "analytics_success_rate.png"
    fig.savefig(paths["analytics_success_rate.png"], dpi=150)
    plt.close(fig)

    # 11) Сводка статистических показателей χ² и p-value
    fig, ax = plt.subplots(figsize=(10, 5))
    names = []
    chi2_df_vals = []
    p_vals = []
    for key, label in [("vertices", "вершины"), ("classes", "категории"),
                       ("priority", "приоритет"), ("deadline", "сроки"),
                       ("resolution", "разрешения"), ("quadrant_grid", "ячейки 3×3")]:
        t = tests.get(key)
        if not t:
            continue
        names.append(label)
        if t.get("applicable") and t.get("df"):
            chi2_df_vals.append(t["statistic"] / t["df"])
            p_vals.append(t.get("p_value"))
        else:
            chi2_df_vals.append(0.0)
            p_vals.append(math.nan)
    if names:
        x = np.arange(len(names))
        colors = []
        for p in p_vals:
            colors.append("#54A24B" if isinstance(p, float) and not math.isnan(p) and p >= 0.05 else "#E45756")
        ax.bar(x, chi2_df_vals, color=colors, edgecolor="black")
        ax.axhline(1.0, color=_COLOR_THEORY, linestyle="--", linewidth=1.2,
                   label="ориентир χ²/df = 1")
        for xi, v, p in zip(x, chi2_df_vals, p_vals):
            p_str = f"p={p:.3g}" if isinstance(p, float) and not math.isnan(p) else "n/a"
            ax.annotate(p_str, (xi, v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title("Статистические показатели согласия: χ²/df и p-value (зелёный — p ≥ 0.05)")
        ax.set_ylabel("χ²/df")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths["analytics_chi2_pvalues.png"] = out / "analytics_chi2_pvalues.png"
    fig.savefig(paths["analytics_chi2_pvalues.png"], dpi=150)
    plt.close(fig)

    return paths


def render_benchmark_charts(rows: list[dict], out_dir: str | Path) -> dict[str, Path]:
    """Графики бенчмарка: время и число проверок в зависимости от числа объектов.

    rows — список словарей с полями: objects_count, combo, duration_s,
    attempts_total, exact_checks_total, peak_mem_mb, ok.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not rows:
        return paths
    combos = sorted({r["combo"] for r in rows})
    counts = sorted({r["objects_count"] for r in rows})

    def _series(combo: str, field: str) -> list[float]:
        by_count = {r["objects_count"]: r.get(field) for r in rows if r["combo"] == combo}
        return [by_count.get(c) for c in counts]

    # 9) Время генерации в зависимости от числа объектов
    fig, ax = plt.subplots(figsize=(8, 5))
    for combo in combos:
        ys = _series(combo, "duration_s")
        ax.plot(counts, ys, marker="o", linewidth=1.6, label=combo)
    ax.set_title("Время генерации в зависимости от числа объектов")
    ax.set_xlabel("число объектов")
    ax.set_ylabel("время, с")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths["benchmark_duration_vs_count.png"] = out / "benchmark_duration_vs_count.png"
    fig.savefig(paths["benchmark_duration_vs_count.png"], dpi=150)
    plt.close(fig)

    # Число точных геометрических проверок
    fig, ax = plt.subplots(figsize=(8, 5))
    for combo in combos:
        ys = _series(combo, "exact_checks_total")
        ax.plot(counts, ys, marker="s", linewidth=1.6, label=combo)
    ax.set_title("Число точных геометрических проверок пар")
    ax.set_xlabel("число объектов")
    ax.set_ylabel("проверок")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths["benchmark_checks_vs_count.png"] = out / "benchmark_checks_vs_count.png"
    fig.savefig(paths["benchmark_checks_vs_count.png"], dpi=150)
    plt.close(fig)

    # Память и попытки
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for combo in combos:
        a1.plot(counts, _series(combo, "peak_mem_mb"), marker="^", linewidth=1.6, label=combo)
        a2.plot(counts, _series(combo, "attempts_total"), marker="o", linewidth=1.6, label=combo)
    a1.set_title("Пиковое потребление памяти")
    a1.set_xlabel("число объектов")
    a1.set_ylabel("память, МБ")
    a1.grid(True, alpha=0.3)
    a1.legend(fontsize=8)
    a2.set_title("Суммарные попытки размещения")
    a2.set_xlabel("число объектов")
    a2.set_ylabel("попыток")
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    paths["benchmark_memory_attempts.png"] = out / "benchmark_memory_attempts.png"
    fig.savefig(paths["benchmark_memory_attempts.png"], dpi=150)
    plt.close(fig)
    return paths


def render_series_comparison(batches: list[tuple[str, dict]], out_dir: str | Path) -> dict[str, Path]:
    """Сравнение результатов серий экспериментов (график 12).

    batches — список пар (метка, batch). Метрики: время, попытки, fill_ratio,
    покрытие, доля успешных прогонов — групповые столбцы по сериям.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not batches:
        return paths
    labels = [b[0] for b in batches]
    summaries = [b[1]["summary"] for b in batches]
    metrics = [
        ("duration_s", "Время прогона, с", lambda s: s["duration_s"]["mean"]),
        ("attempts_total", "Попыток размещения", lambda s: s["attempts_total"]["mean"]),
        ("fill_ratio", "fill_ratio", lambda s: s["fill_ratio"]["mean"]),
        ("coverage", "Покрытие объектов", lambda s: s["coverage_fraction"]["mean"]),
        ("success", "Доля успешных", lambda s: s["n_ok"] / max(1, s["n_runs"])),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    colors = plt.cm.tab10.colors
    for ax, (key, title, getter) in zip(axes, metrics):
        vals = [getter(s) for s in summaries]
        bars = ax.bar(range(len(labels)), vals, color=[colors[i % 10] for i in range(len(labels))],
                      edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25 + 1e-12)
    fig.suptitle("Сравнение серий экспериментов")
    fig.tight_layout()
    paths["analytics_series_compare.png"] = out / "analytics_series_compare.png"
    fig.savefig(paths["analytics_series_compare.png"], dpi=150)
    plt.close(fig)
    return paths
