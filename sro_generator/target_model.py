"""Модель объектов съёмки: четыре закона пространственного размещения.

Модуль реализует математические модели пространственного распределения целевых
объектов ДЗЗ внутри района (в локальной метрической СК, см. ``geometry.LocalCRS``)
и инструментарий их статистической проверки.

Законы размещения (генерируются центры, форма строится общим геометрическим
ядром ``geometry.generate_polygon_shape``; соблюдаются границы района с отступом,
запрет пересечений по площади и минимальное расстояние между полигонами —
семантика :func:`placement.pair_placement_ok`):

1. ``uniform`` — равномерно-случайное: центры — независимые равномерно
   распределённые точки в допустимой области (однородный пуассоновский процесс
   с фиксированным числом точек, биномиальный процесс).
2. ``cluster`` — групповое (кластерное): K центров групп — равномерно в области;
   центр каждого объекта — случайный выбор группы (равномерно) и нормальное
   (гауссово) отклонение от её центра с СКО ``cluster_sigma_m``.
3. ``regular`` — регулярное: центры — узлы квадратной решётки с шагом,
   подобранным по площади области (число узлов ≈ числу объектов), со случайным
   прореживанием лишних узлов и равномерным «дрожанием» узлов в пределах
   ``jitter_fraction`` · шаг/2 по каждой оси; минимальные расстояния обеспечивает
   решётка и общие ограничения размещения.
4. ``linear`` — линейное: L прямолинейных структур (хорды района со случайной
   точкой и азимутом); центр объекта — равномерная точка вдоль хорды случайно
   выбранной структуры плюс нормальное поперечное отклонение ``line_sigma_m``.

Статистическая проверка (классическая школа статистических испытаний):
для каждой независимой реализации вычисляются пространственные характеристики
(среднее расстояние до ближайшего соседа, индекс Кларка–Эванса R с z-статистикой,
индекс дисперсии квадрантного счёта I = s²/x̄ с χ²-критерием, средняя результирующая
длина направлений к ближайшему соседу — мера анизотропии, доля покрытия района).
По N независимым реализациям строятся накопленные оценки M*(n), D̃(n) (деление
на n−1), σ̃(n), точность ε(β,n) = t(β,n−1)·σ̃/√n и доверительные интервалы
(функции ``monte_carlo.classical_series`` / ``required_trials``).

Оговорка о зависимости: объекты внутри одной реализации зависимы (условия
непересечения и общая структура модели), поэтому одновыборочные критерии
(Кларка–Эванса, χ² квадрантов) применяются как приближённые диагностики
конкретной реализации, а статистические выводы о законе делаются по выборке
независимых реализаций (между реализациями зависимости нет).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from . import __version__
from .geometry import generate_polygon_shape, is_proper_polygon
from .monte_carlo import BETAS, _z_norm, classical_series, required_trials, t_crit
from .placement import PlacementError, _sample_point_in_polygon, pair_placement_ok

__all__ = [
    "MODELS",
    "SpatialModelParams",
    "TargetObject",
    "TargetModelResult",
    "generate_target_objects",
    "nearest_neighbor_stats",
    "quadrant_stats",
    "chi2_quantile",
    "realization_metrics",
    "run_target_model_study",
    "write_target_csv",
    "write_target_json",
    "render_target_charts",
    "export_target_geojson",
    "nn_distance_sample",
    "collect_nn_samples",
    "sample_skewness",
    "rayleigh_cdf",
    "render_statistics_charts",
]

MODELS = ("uniform", "cluster", "regular", "linear")

MODEL_TITLES_RU = {
    "uniform": "Равномерно-случайное",
    "cluster": "Групповое (кластерное)",
    "regular": "Регулярное",
    "linear": "Линейное",
}


@dataclass
class SpatialModelParams:
    """Параметры модели объектов съёмки.

    Общие (форма и ограничения):
    - ``count`` — число объектов;
    - ``vertices_min``/``vertices_max`` — число вершин контура (3…10);
    - ``area_min_m2``/``area_max_m2`` — диапазон площадей объектов;
    - ``irregularity`` — нерегулярность формы (0…1);
    - ``max_aspect_ratio`` — предельное отношение сторон (больше 1);
    - ``boundary_margin_m`` — отступ полигонов от границы района;
    - ``min_distance_m`` — минимальное расстояние между полигонами (≥ 0);
    - ``max_attempts_per_object`` — предел попыток размещения одного объекта;
    - ``allow_overlap`` — разрешить пересечения полигонов по площади
      (False по умолчанию; True используется для полей препятствий-облаков,
      которые физически могут сливаться; см. ``joint_model``).

    Закон размещения — ``model`` ∈ {uniform, cluster, regular, linear}:
    - ``cluster``: ``n_clusters`` — число групп, ``cluster_sigma_m`` — СКО
      гауссова отклонения центра объекта от центра группы;
    - ``regular``: ``jitter_fraction`` — доля полушага решётки для равномерного
      дрожания узлов (0 — строгая решётка, 1 — дрожание на полный полушаг);
    - ``linear``: ``n_lines`` — число линейных структур, ``line_sigma_m`` — СКО
      поперечного отклонения от структуры.
    """

    model: str = "uniform"
    count: int = 60
    vertices_min: int = 3
    vertices_max: int = 10
    area_min_m2: float = 50_000.0
    area_max_m2: float = 500_000.0
    irregularity: float = 0.35
    max_aspect_ratio: float = 2.5
    rotation_min_deg: float = 0.0
    rotation_max_deg: float = 360.0
    boundary_margin_m: float = 100.0
    min_distance_m: float = 0.0
    max_attempts_per_object: int = 10_000
    allow_overlap: bool = False
    n_clusters: int = 5
    cluster_sigma_m: float = 1_500.0
    jitter_fraction: float = 0.6
    n_lines: int = 4
    line_sigma_m: float = 300.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"target_model.model: допустимые значения {MODELS}.")
        if self.count < 1:
            raise ValueError("target_model.count: число объектов должно быть не меньше 1.")
        if not (3 <= self.vertices_min <= self.vertices_max <= 10):
            raise ValueError("target_model.vertices_min/max: число вершин 3…10, мин ≤ макс.")
        if not (0.0 < self.area_min_m2 <= self.area_max_m2):
            raise ValueError("target_model.area_min/max_m2: площади положительные, мин ≤ макс.")
        if not (0.0 <= self.irregularity <= 1.0):
            raise ValueError("target_model.irregularity: нерегулярность от 0 до 1.")
        if self.max_aspect_ratio <= 1.0:
            raise ValueError("target_model.max_aspect_ratio: должно быть больше 1.")
        if self.rotation_min_deg > self.rotation_max_deg:
            raise ValueError("target_model.rotation_min_deg: минимум поворота больше максимума.")
        if self.boundary_margin_m < 0.0 or self.min_distance_m < 0.0:
            raise ValueError("target_model: отступы и расстояния не могут быть отрицательными.")
        if self.max_attempts_per_object < 1:
            raise ValueError("target_model.max_attempts_per_object: должно быть положительным.")
        if self.n_clusters < 1:
            raise ValueError("target_model.n_clusters: число групп должно быть не меньше 1.")
        if self.cluster_sigma_m <= 0.0:
            raise ValueError("target_model.cluster_sigma_m: СКО группы должно быть положительным.")
        if not (0.0 <= self.jitter_fraction <= 1.0):
            raise ValueError("target_model.jitter_fraction: дрожание решётки от 0 до 1.")
        if self.n_lines < 1:
            raise ValueError("target_model.n_lines: число линейных структур не меньше 1.")
        if self.line_sigma_m < 0.0:
            raise ValueError("target_model.line_sigma_m: СКО отклонения не может быть отрицательным.")


@dataclass
class TargetObject:
    """Размещённый целевой объект: геометрия (локальные метры) и характеристики."""

    object_id: str
    geometry: Polygon
    center: Point
    area_m2: float
    vertex_count: int
    rotation_deg: float
    group_id: int | None = None   # номер группы (cluster) или структуры (linear)


@dataclass
class TargetModelResult:
    """Результат размещения: объекты и параметры реализации модели."""

    params: SpatialModelParams
    district: Polygon
    objects: list[TargetObject]
    cluster_centers: list[Point] = field(default_factory=list)
    line_chords: list[LineString] = field(default_factory=list)
    attempts_total: int = 0
    warnings: list[str] = field(default_factory=list)


def _make_shape(rng: np.random.Generator, p: SpatialModelParams) -> tuple[Polygon, int, float]:
    """Случайная форма объекта (общее геометрическое ядро)."""
    n = int(rng.integers(p.vertices_min, p.vertices_max + 1))
    target_area = float(rng.uniform(p.area_min_m2, p.area_max_m2))
    rot = float(rng.uniform(p.rotation_min_deg, p.rotation_max_deg))
    for _ in range(200):
        try:
            poly = generate_polygon_shape(rng, n, target_area, p.irregularity, rot, p.max_aspect_ratio)
        except ValueError:
            continue
        if is_proper_polygon(poly, n):
            return poly, n, rot
    raise PlacementError("target_model: не удалось построить корректную форму объекта за 200 попыток.")


def _translate(poly: Polygon, cx: float, cy: float) -> Polygon:
    return Polygon([(x + cx, y + cy) for x, y in poly.exterior.coords[:-1]])


# ---------------------------------------------------------------------------
# Генераторы кандидатов центров для каждого закона
# ---------------------------------------------------------------------------

def _uniform_candidates(rng, inner, p, n):
    """Равномерно-случайные центры в допустимой области (бесконечный поток)."""
    while True:
        yield _sample_point_in_polygon(rng, inner), None


def _regular_lattice_nodes(rng, inner, p, n):
    """Узлы квадратной решётки с дрожанием; шаг подбирается по площади области."""
    minx, miny, maxx, maxy = inner.bounds
    s = math.sqrt(inner.area / max(n, 1))
    nodes: list[tuple[float, float]] = []
    for _ in range(200):
        nx = max(1, int(math.ceil((maxx - minx) / s)))
        ny = max(1, int(math.ceil((maxy - miny) / s)))
        nodes = [
            (minx + (i + 0.5) * (maxx - minx) / nx, miny + (j + 0.5) * (maxy - miny) / ny)
            for i in range(nx) for j in range(ny)
            if inner.covers(Point(minx + (i + 0.5) * (maxx - minx) / nx,
                                  miny + (j + 0.5) * (maxy - miny) / ny))
        ]
        if len(nodes) >= n:
            break
        s *= 0.97
    if len(nodes) < n:
        raise PlacementError(
            f"target_model.regular: удалось получить только {len(nodes)} узлов решётки "
            f"для {n} объектов. Уменьшите число объектов или отступы."
        )
    order = rng.permutation(len(nodes))
    picked = [nodes[int(i)] for i in order[:n]]
    half = 0.5 * p.jitter_fraction * s
    # бесконечный поток: узлы циклируются, дрожание каждый раз новое
    while True:
        for x, y in picked:
            pt = Point(x, y)
            for _ in range(20):
                cand = Point(float(rng.uniform(x - half, x + half)), float(rng.uniform(y - half, y + half)))
                if inner.covers(cand):
                    pt = cand
                    break
            yield pt, None


def generate_target_objects(
    params: SpatialModelParams,
    district: Polygon,
    rng: np.random.Generator,
) -> TargetModelResult:
    """Разместить ``params.count`` объектов в районе по выбранному закону.

    Центры — из генератора кандидатов закона; форма — общее геометрическое
    ядро; ограничения: полигон внутри района с отступом, нет пересечений по
    площади, расстояние между полигонами не меньше ``min_distance_m``.
    При невозможности разместить объект за ``max_attempts_per_object`` попыток —
    PlacementError. Одинаковые params, district и seed rng дают одинаковый
    результат. Структура модели (центры групп для cluster, хорды для linear)
    сохраняется в результате.
    """
    p = params
    inner = district.buffer(-p.boundary_margin_m)
    if inner.is_empty or inner.area <= 0.0:
        raise PlacementError(
            "target_model.boundary_margin_m: отступ от границы слишком велик для заданного района."
        )
    result = TargetModelResult(params=p, district=district, objects=[])
    placed: list[TargetObject] = []

    if p.model == "uniform":
        candidates = _uniform_candidates(rng, inner, p, p.count)
    elif p.model == "cluster":
        centers = [_sample_point_in_polygon(rng, inner) for _ in range(p.n_clusters)]
        result.cluster_centers = centers

        def cluster_gen():
            while True:
                g = int(rng.integers(0, len(centers)))
                c = centers[g]
                yield Point(float(rng.normal(c.x, p.cluster_sigma_m)),
                            float(rng.normal(c.y, p.cluster_sigma_m))), g

        candidates = cluster_gen()
    elif p.model == "regular":
        candidates = _regular_lattice_nodes(rng, inner, p, p.count)
    else:  # linear
        minx, miny, maxx, maxy = inner.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        chords: list[LineString] = []
        for _ in range(p.n_lines):
            base = _sample_point_in_polygon(rng, inner)
            az = float(rng.uniform(0.0, math.pi))
            dx, dy = math.cos(az), math.sin(az)
            line = LineString([(base.x - diag * dx, base.y - diag * dy),
                               (base.x + diag * dx, base.y + diag * dy)])
            chord = line.intersection(inner)
            if chord.is_empty or chord.length <= 0.0:
                continue
            if chord.geom_type == "MultiLineString":
                chord = max(chord.geoms, key=lambda g: g.length)
            if chord.geom_type == "LineString" and chord.length > 0.0:
                chords.append(chord)
        if not chords:
            raise PlacementError("target_model.linear: не удалось построить ни одной хорды района.")
        result.line_chords = chords

        def linear_gen():
            while True:
                li = int(rng.integers(0, len(chords)))
                chord = chords[li]
                pt0 = chord.interpolate(float(rng.random()), normalized=True)
                coords = list(chord.coords)
                az = math.atan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0])
                off = float(rng.normal(0.0, p.line_sigma_m))
                yield Point(pt0.x - math.sin(az) * off, pt0.y + math.cos(az) * off), li

        candidates = linear_gen()

    for k in range(p.count):
        object_id = f"TGT-{k + 1:03d}"
        poly, nv, rot = _make_shape(rng, p)
        ok = False
        attempts = 0
        while attempts < p.max_attempts_per_object and not ok:
            attempts += 1
            try:
                pt, group_id = next(candidates)
            except StopIteration:
                raise PlacementError(
                    f"target_model: генератор кандидатов исчерпан на объекте {object_id}."
                )
            if not inner.covers(pt):
                continue
            shifted = _translate(poly, pt.x, pt.y)
            if not inner.covers(shifted):
                continue
            conflict = False
            for other in placed:
                if not pair_placement_ok(shifted, other.geometry, p.min_distance_m,
                                         p.allow_overlap, 1.0 if p.allow_overlap else 0.0):
                    conflict = True
                    break
            if conflict:
                continue
            placed.append(TargetObject(
                object_id=object_id,
                geometry=shifted,
                center=pt,
                area_m2=float(shifted.area),
                vertex_count=nv,
                rotation_deg=rot,
                group_id=group_id,
            ))
            ok = True
        result.attempts_total += attempts
        if not ok:
            raise PlacementError(
                f"target_model: объект {object_id} не размещён за "
                f"{p.max_attempts_per_object} попыток. Уменьшите число объектов, "
                "площади или отступы либо увеличьте район."
            )

    result.objects = placed
    return result


# ---------------------------------------------------------------------------
# Пространственные статистики одной реализации
# ---------------------------------------------------------------------------

def nearest_neighbor_stats(centers_xy: np.ndarray, district_area_m2: float, boundary=None) -> dict:
    """Статистики ближайшего соседа: среднее расстояние d̄, индекс Кларка–Эванса
    R = d̄ / (0,5/√λ), z-статистика (SE по Доннелли) и средняя результирующая
    длина направлений к ближайшему соседу (осевая, по удвоенным углам;
    мера анизотропии, 0…1).

    λ = n / A — интенсивность процесса. Критерий приближённый: краевые эффекты
    не корректируются, объекты одной реализации зависимы.
    """
    pts = np.asarray(centers_xy, dtype=float)
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError("nearest_neighbor_stats: нужно не менее 2 точек.")
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(axis=1)
    d = dist[np.arange(n), nn]
    lam = n / district_area_m2
    d_exp = 0.5 / math.sqrt(lam)
    se = 0.26136 / math.sqrt(n * lam)  # SE среднего расстояния (Donnelly, 1978)
    d_bar = float(d.mean())
    n_eff = n
    if boundary is not None:
        # поправка на краевые эффекты (метод граничной полосы): в среднее d̄
        # входят только объекты, чей центр удалён от границы района не меньше
        # чем на d̄ (их ближайший сосед гарантированно внутри района)
        from shapely.geometry import Point as _Pt

        edge = np.array([boundary.exterior.distance(_Pt(x, y)) for x, y in pts], dtype=float)
        keep = edge >= d_bar
        if int(keep.sum()) >= 2:
            d_bar = float(d[keep].mean())
            n_eff = int(keep.sum())
            se = 0.26136 / math.sqrt(n_eff * lam)
    bearings = np.arctan2(pts[nn, 1] - pts[:, 1], pts[nn, 0] - pts[:, 0])
    # направление к ближайшему соседу осевое (θ ≡ θ+π): результирующая по удвоенным углам
    bearing_r = float(abs(np.mean(np.exp(2j * bearings))))
    return {
        "n": n,
        "nn_mean_m": d_bar,
        "nn_std_m": float(d.std(ddof=1)) if n > 2 else math.nan,
        "nn_expected_m": d_exp,
        "nn_n_eff": n_eff,
        "ce_r": d_bar / d_exp,
        "ce_z": (d_bar - d_exp) / se,
        "nn_bearing_r": bearing_r,
    }


def chi2_quantile(prob: float, df: int) -> float:
    """Квантиль χ² с df степенями свободы (аппроксимация Уилсона–Хилферти).

    Для df ≥ 10 погрешность мала; используется для двустороннего χ²-критерия
    квадрантного счёта (df = число ячеек − 1).
    """
    if df < 1:
        raise ValueError("chi2_quantile: df должно быть не меньше 1.")
    if not (0.0 < prob < 1.0):
        raise ValueError("chi2_quantile: вероятность должна быть в (0, 1).")
    if prob == 0.5:
        z = 0.0
    elif prob > 0.5:
        z = _z_norm(2.0 * prob - 1.0)   # Φ⁻¹(prob) через двусторонний квантиль
    else:
        z = -_z_norm(2.0 * (1.0 - prob) - 1.0)
    h = 2.0 / (9.0 * df)
    return df * (1.0 - h + z * math.sqrt(h)) ** 3


def quadrant_stats(
    centers_xy: np.ndarray,
    district: Polygon,
    grid: int = 5,
) -> dict:
    """Квадрантный анализ: сетка grid×grid над районом, счёт точек в ячейках,
    индекс дисперсии I = s²/x̄ (s² — исправленная выборочная дисперсия счёта,
    деление на m−1) и χ² = (m−1)·I ~ χ²(m−1) при гипотезе случайного
    равномерного размещения (CSR).

    I ≈ 1 — случайное, I < 1 — регулярное, I > 1 — групповое размещение.
    Для повёрнутого района ячейки угловой части покрывают район частично —
    критерий приближённый; исследовательский стенд использует осевой район.
    """
    pts = np.asarray(centers_xy, dtype=float)
    minx, miny, maxx, maxy = district.bounds
    ix = np.clip(((pts[:, 0] - minx) / (maxx - minx) * grid).astype(int), 0, grid - 1)
    iy = np.clip(((pts[:, 1] - miny) / (maxy - miny) * grid).astype(int), 0, grid - 1)
    counts = np.zeros((grid, grid), dtype=float)
    np.add.at(counts, (iy, ix), 1.0)
    flat = counts.ravel()
    m = flat.size
    mean = float(flat.mean())
    var = float(flat.var(ddof=1)) if m > 1 else 0.0
    disp = var / mean if mean > 0.0 else math.nan
    chi2 = (m - 1) * disp if not math.isnan(disp) else math.nan
    lo = chi2_quantile(0.025, m - 1)
    hi = chi2_quantile(0.975, m - 1)
    return {
        "grid": grid,
        "cells": m,
        "mean_count": mean,
        "var_count": var,
        "dispersion_index": disp,
        "chi2": chi2,
        "chi2_low_crit": lo,
        "chi2_high_crit": hi,
        "csr_rejected": bool(not math.isnan(chi2) and (chi2 < lo or chi2 > hi)),
        "pattern": ("regular" if chi2 < lo else ("clustered" if chi2 > hi else "random"))
        if not math.isnan(chi2) else "undefined",
    }


def realization_metrics(result: TargetModelResult, grid: int = 5) -> dict:
    """Все пространственные характеристики одной реализации модели."""
    objs = result.objects
    centers = np.array([[o.center.x, o.center.y] for o in objs], dtype=float)
    area_district = float(result.district.area)
    from shapely.ops import unary_union

    union_area = float(unary_union([o.geometry for o in objs]).area)
    nn = nearest_neighbor_stats(centers, area_district, boundary=result.district)
    quad = quadrant_stats(centers, result.district, grid=grid)
    areas = np.array([o.area_m2 for o in objs], dtype=float)
    return {
        "model": result.params.model,
        "n_objects": len(objs),
        "density_per_km2": len(objs) / (area_district / 1e6),
        "district_area_m2": area_district,
        "objects_area_mean_m2": float(areas.mean()),
        "objects_area_fraction": union_area / area_district,
        **nn,
        "dispersion_index": quad["dispersion_index"],
        **{f"quad_{k}": v for k, v in quad.items() if k != "dispersion_index"},
    }


# ---------------------------------------------------------------------------
# Статистическое исследование: N независимых реализаций каждого закона
# ---------------------------------------------------------------------------

_STUDY_METRICS = (
    "nn_mean_m",
    "ce_r",
    "dispersion_index",
    "nn_bearing_r",
    "objects_area_fraction",
)

_STUDY_METRIC_TITLES_RU = {
    "nn_mean_m": "Среднее расстояние до ближайшего соседа, м",
    "ce_r": "Индекс Кларка–Эванса R",
    "dispersion_index": "Индекс дисперсии квадрантного счёта I",
    "nn_bearing_r": "Результирующая длина направлений к ближайшему соседу",
    "objects_area_fraction": "Доля площади района, занятая объектами",
}


def run_target_model_study(
    params: SpatialModelParams,
    district: Polygon,
    realizations: int = 200,
    seed_start: int = 0,
    beta: float = 0.95,
    grid: int = 5,
    models: tuple[str, ...] = MODELS,
    progress=None,
) -> dict:
    """Независимые статистические испытания законов размещения.

    Для каждого закона из ``models`` — ``realizations`` независимых реализаций
    (seed_start + i); параметры модели и района во всех реализациях одинаковы.
    По накопленным выборкам каждой метрики строятся M*(n), D̃(n), σ̃(n),
    ε(β,n) и доверительные интервалы (``classical_series``); фиксируется доля
    реализаций, отвергающих гипотезу CSR по χ²-квадрантному критерию и по
    z-критерию Кларка–Эванса (|z| > 1.96).
    """
    if realizations < 3:
        raise ValueError("run_target_model_study: нужно не менее 3 реализаций.")
    if beta not in BETAS:
        raise ValueError(f"run_target_model_study: β должна быть одной из {BETAS}.")

    per_model: dict[str, dict] = {}
    done = 0
    total = realizations * len(models)
    warnings: list[str] = []
    for model in models:
        p = SpatialModelParams(**{**asdict(params), "model": model})
        runs: list[dict] = []
        retries_total = 0
        for i in range(realizations):
            # при невозможности размещения (перенасыщенные группы) реализация
            # детерминированно перегенерируется со следующим seed; число
            # повторов фиксируется — условное распределение реализуемых
            # реализаций, как в исследовании Монте-Карло
            seed_used = seed_start + i
            for _ in range(100):
                rng = np.random.default_rng(np.random.SeedSequence([seed_used, 20261126, len(model)]))
                try:
                    result = generate_target_objects(p, district, rng)
                    break
                except PlacementError:
                    seed_used += 1
                    retries_total += 1
            else:
                raise PlacementError(
                    f"run_target_model_study: закон «{model}», реализация {i}: "
                    "не удалось получить реализуемую конфигурацию за 100 попыток."
                )
            metrics = realization_metrics(result, grid=grid)
            metrics["seed_used"] = seed_used
            runs.append(metrics)
            done += 1
            if progress:
                progress(done, total)
        if retries_total:
            warnings.append(
                f"Закон «{model}»: {retries_total} повторов генерации реализации "
                "(размещение из исходного seed невозможно): использовано условное "
                "распределение по реализуемым реализациям."
            )
        classical: dict[str, list[dict]] = {}
        for metric in _STUDY_METRICS:
            classical[metric] = classical_series([r[metric] for r in runs], beta=beta)
        z_reject = sum(1 for r in runs if abs(r["ce_z"]) > 1.96)
        chi2_reject = sum(1 for r in runs if r["quad_csr_rejected"])
        per_model[model] = {
            "params": asdict(p),
            "runs": runs,
            "classical": classical,
            "ce_z_reject_share": z_reject / realizations,
            "csr_chi2_reject_share": chi2_reject / realizations,
        }

    return {
        "version": __version__,
        "mode": "target_object_distribution",
        "realizations": realizations,
        "seed_start": seed_start,
        "beta": beta,
        "grid": grid,
        "models": list(models),
        "district_area_m2": float(district.area),
        "metrics": list(_STUDY_METRICS),
        "metric_titles_ru": dict(_STUDY_METRIC_TITLES_RU),
        "per_model": per_model,
        "warnings": warnings,
        "note": (
            "Объекты внутри одной реализации зависимы; одновыборочные критерии "
            "(Кларка–Эванса, χ² квадрантов) — приближённые диагностики реализации. "
            "Выводы о законе — по выборке независимых реализаций."
        ),
    }


# ---------------------------------------------------------------------------
# Экспорт: CSV, JSON, GeoJSON, PNG
# ---------------------------------------------------------------------------

def write_target_csv(study: dict, out_dir: str | Path) -> dict[str, Path]:
    """CSV: реализации по моделям и сводка итоговых оценок (M*, D̃, σ̃, ε, ДИ)."""
    import csv
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    runs_path = out / "target_runs.csv"
    fields = [
        "model", "realization", "n_objects", "density_per_km2", "objects_area_mean_m2",
        "objects_area_fraction", "nn_mean_m", "nn_std_m", "nn_expected_m",
        "ce_r", "ce_z", "nn_bearing_r",
        "quad_mean_count", "quad_var_count", "dispersion_index",
        "quad_chi2", "quad_csr_rejected", "quad_pattern",
    ]
    with runs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in study["models"]:
            for i, r in enumerate(study["per_model"][model]["runs"]):
                writer.writerow({
                    "model": model,
                    "realization": i,
                    **{k: (round(r[k], 8) if isinstance(r.get(k), float) else r.get(k))
                       for k in fields if k not in ("model", "realization")},
                })
    paths["target_runs.csv"] = runs_path

    summ_path = out / "target_summary.csv"
    with summ_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["model", "metric", "n", "m_star", "d_tilde", "sigma_tilde",
                  "epsilon", "ci_low", "ci_high", "ce_z_reject_share", "csr_chi2_reject_share"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in study["models"]:
            pm = study["per_model"][model]
            for metric in study["metrics"]:
                last = pm["classical"][metric][-1]
                writer.writerow({
                    "model": model,
                    "metric": metric,
                    "n": last["n"],
                    "m_star": round(last["m_star"], 8),
                    "d_tilde": round(last["d_tilde"], 10),
                    "sigma_tilde": round(last["sigma_tilde"], 8),
                    "epsilon": round(last["epsilon"], 8),
                    "ci_low": round(last["ci_low"], 8),
                    "ci_high": round(last["ci_high"], 8),
                    "ce_z_reject_share": round(pm["ce_z_reject_share"], 6),
                    "csr_chi2_reject_share": round(pm["csr_chi2_reject_share"], 6),
                })
    paths["target_summary.csv"] = summ_path
    return paths


def write_target_json(study: dict, out_dir: str | Path) -> Path:
    """Записать всё исследование в target_distribution.json."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "target_distribution.json"

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(f"Не сериализуется в JSON: {o.__class__.__name__}")

    path.write_text(json.dumps(study, ensure_ascii=False, indent=2, default=_default),
                    encoding="utf-8")
    return path


def export_target_geojson(result: TargetModelResult, local) -> dict:
    """GeoJSON (FeatureCollection) объектов модели в WGS 84 (lon, lat).

    ``local`` — LocalCRS для перевода координат. Свойства: object_id, model,
    group_id, area_m2, vertex_count, rotation_deg, центр в WGS 84 и локальных
    метрах, параметры модели в свойствах коллекции не дублируются.
    """
    from shapely.geometry import mapping

    features = []
    for o in result.objects:
        clon, clat = local.to_lonlat(o.center.x, o.center.y)
        features.append({
            "type": "Feature",
            "geometry": mapping(local.polygon_to_lonlat(o.geometry)),
            "properties": {
                "object_id": o.object_id,
                "model": result.params.model,
                "group_id": o.group_id,
                "area_m2": round(o.area_m2, 3),
                "vertex_count": o.vertex_count,
                "rotation_deg": round(o.rotation_deg, 6),
                "center_lat": clat,
                "center_lon": clon,
                "center_x_m": round(o.center.x, 6),
                "center_y_m": round(o.center.y, 6),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def render_target_charts(study: dict, examples: dict, out_dir: str | Path) -> dict[str, Path]:
    """Научные рисунки: примеры реализаций, сходимость M*(n), гистограммы и
    сравнительные диаграммы пространственных индексов."""
    import matplotlib
    from pathlib import Path

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    models = study["models"]
    colors = plt.cm.tab10.colors
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    beta = study["beta"]

    # 1) примеры реализаций (по одной на закон)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.6))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        res = examples[model]
        dx, dy = res.district.exterior.xy
        ax.plot(dx, dy, "k-", linewidth=1.2)
        for chord in res.line_chords:
            cx, cy = chord.xy
            ax.plot(cx, cy, color="0.5", linestyle="--", linewidth=1.0)
        for c in res.cluster_centers:
            ax.plot(c.x, c.y, marker="x", color="red", markersize=9, markeredgewidth=2.0)
        for o in res.objects:
            x, y = o.geometry.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True,
                                    facecolor=model_colors[model], edgecolor="black",
                                    linewidth=0.5, alpha=0.75))
        ax.set_aspect("equal")
        ax.set_title(MODEL_TITLES_RU[model], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Примеры реализаций законов размещения целевых объектов "
                 f"(n = {study['per_model'][models[0]]['runs'][0]['n_objects']} объектов)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "target_examples.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # 2) сходимость M*(n) среднего расстояния до ближайшего соседа
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in models:
        series = study["per_model"][model]["classical"]["nn_mean_m"]
        ns = [r["n"] for r in series]
        m_star = [r["m_star"] for r in series]
        lo = [r["ci_low"] for r in series]
        hi = [r["ci_high"] for r in series]
        ax.fill_between(ns, lo, hi, color=model_colors[model], alpha=0.2)
        ax.plot(ns, m_star, color=model_colors[model], linewidth=1.8,
                label=MODEL_TITLES_RU[model])
    ax.set_title(f"Сходимость статистического математического ожидания M*(n)\n"
                 f"среднего расстояния до ближайшего соседа (лента — ДИ, β = {beta:.2f})",
                 fontsize=11)
    ax.set_xlabel("объём испытаний n (число независимых реализаций)")
    ax.set_ylabel("M*(n), м")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out / "target_nn_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # 3) гистограммы индексов R и I по моделям
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, metric, title in (
        (axes[0], "ce_r", "Индекс Кларка–Эванса R\n(R < 1 — группирование, R ≈ 1 — случайное, R > 1 — регулярное)"),
        (axes[1], "dispersion_index", "Индекс дисперсии квадрантного счёта I\n(I < 1 — регулярное, I ≈ 1 — случайное, I > 1 — групповое)"),
    ):
        for model in models:
            vals = [r[metric] for r in study["per_model"][model]["runs"]]
            ax.hist(vals, bins=24, alpha=0.5, color=model_colors[model],
                    label=MODEL_TITLES_RU[model], edgecolor="black", linewidth=0.3)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("значение индекса")
        ax.set_ylabel("число реализаций")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "target_index_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    # 4) сравнительные диаграммы: R, I и анизотропия с 95 %-ми ДИ среднего
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric, title in (
        (axes[0], "ce_r", "Индекс Кларка–Эванса R"),
        (axes[1], "dispersion_index", "Индекс дисперсии I"),
        (axes[2], "nn_bearing_r", "Анизотропия направлений R_b"),
    ):
        means, halves, labels, cols = [], [], [], []
        for model in models:
            last = study["per_model"][model]["classical"][metric][-1]
            means.append(last["m_star"])
            halves.append(last["epsilon"])
            labels.append(MODEL_TITLES_RU[model])
            cols.append(model_colors[model])
        x = np.arange(len(models))
        ax.bar(x, means, yerr=halves, capsize=5, color=cols, alpha=0.8,
               edgecolor="black", linewidth=0.5)
        ax.axhline(1.0 if metric != "nn_bearing_r" else 0.0,
                   color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=12)
        ax.set_title(f"{title}\n(M* ± ε, β = {beta:.2f}, n = {study['realizations']})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "target_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths[p.name] = p

    return paths


# ---------------------------------------------------------------------------
# Статистическое исследование законов: выборки расстояний до ближайшего
# соседа, имитационный эталон, шесть обязательных графиков (вкладка 4)
# ---------------------------------------------------------------------------

def nn_distance_sample(result: TargetModelResult) -> np.ndarray:
    """Выборка расстояний до ближайшего соседа одной реализации с поправкой
    на краевые эффекты (метод граничной полосы, как в nearest_neighbor_stats):
    включаются только объекты, чей центр удалён от границы района не меньше
    чем на среднее расстояние d̄ реализации. Это характеристики отдельных
    объектов внутри реализации (зависимые), а не независимые испытания —
    использовать для формы распределения F*(r), не для ДИ среднего."""
    objs = result.objects
    if len(objs) < 2:
        return np.empty(0, dtype=float)
    pts = np.array([[o.center.x, o.center.y] for o in objs], dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    d = dist.min(axis=1)
    edge = np.array(
        [result.district.exterior.distance(Point(x, y)) for x, y in pts], dtype=float)
    keep = edge >= float(d.mean())
    if int(keep.sum()) < 2:
        return d
    return d[keep]


def collect_nn_samples(
    params: SpatialModelParams,
    district: Polygon,
    models: tuple[str, ...] = MODELS,
    realizations: int = 100,
    seed_start: int = 0,
    progress=None,
) -> dict[str, np.ndarray]:
    """Пул расстояний до ближайшего соседа по независимым реализациям каждого
    закона (для эмпирической функции распределения F*(r)).

    Та же детерминированная схема seed, что в run_target_model_study
    (seed_start + i, детерминированная перегенерация нереализуемых конфигураций),
    поэтому при совпадающих параметрах выборки соответствуют сохранённым
    реализациям исследования. Независимый имитационный эталон — тот же вызов
    с другим диапазоном seed (например, seed_start = 100 000).
    """
    if realizations < 1:
        raise ValueError("collect_nn_samples: число реализаций должно быть не меньше 1.")
    out: dict[str, np.ndarray] = {}
    done = 0
    total = realizations * len(models)
    for model in models:
        p = SpatialModelParams(**{**asdict(params), "model": model})
        pooled: list[np.ndarray] = []
        for i in range(realizations):
            seed_used = seed_start + i
            for _ in range(100):
                rng = np.random.default_rng(np.random.SeedSequence([seed_used, 20261126, len(model)]))
                try:
                    result = generate_target_objects(p, district, rng)
                    break
                except PlacementError:
                    seed_used += 1
            else:
                raise PlacementError(
                    f"collect_nn_samples: закон «{model}», реализация {i}: "
                    "не удалось получить реализуемую конфигурацию за 100 попыток."
                )
            pooled.append(nn_distance_sample(result))
            done += 1
            if progress:
                progress(done, total)
        out[model] = np.concatenate(pooled) if pooled else np.empty(0, dtype=float)
    return out


def sample_skewness(values) -> float:
    """Выборочный коэффициент асимметрии (Фишер–Пирсон, g1) — индикатор
    отклонения распределения реализаций от нормального при оценке
    применимости t-интервалов для малых выборок."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 3:
        return math.nan
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    if s <= 0.0:
        return 0.0
    return float(((arr - m) ** 3).mean() / s ** 3 * n ** 2 / ((n - 1) * (n - 2)))


def rayleigh_cdf(r: np.ndarray, lam: float) -> np.ndarray:
    """Теоретическая функция распределения расстояния до ближайшего соседа
    для однородного пуассоновского процесса (CSR): F(r) = 1 − exp(−πλr²).

    Аналитический эталон не учитывает запрет пересечений полигонов и краевые
    эффекты; для законов с такими ограничениями эталоном служит независимая
    имитационная выборка (collect_nn_samples с другим диапазоном seed).
    """
    r = np.asarray(r, dtype=float)
    return 1.0 - np.exp(-math.pi * lam * r * r)


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, v.size + 1, dtype=float) / v.size


def render_statistics_charts(
    study: dict,
    examples: dict,
    nn_empirical: dict[str, np.ndarray],
    nn_reference: dict[str, np.ndarray] | None,
    selected_model: str,
    metric: str,
    beta: float,
    epsilon0: float,
    out_dir: str | Path,
    formats: tuple[str, ...] = ("png",),
) -> dict[str, list[Path]]:
    """Шесть обязательных научных графиков исследования законов размещения.

    1 — реализации четырёх законов (единый масштаб, район и число объектов);
    2 — эмпирические F*(r) расстояний до ближайшего соседа: по каждому закону
        эмпирика (испытания), независимый имитационный эталон и аналитическая
        кривая Рэлея для CSR;
    3 — сравнение законов по R, I и R_b с доверительными границами (M* ± ε);
    4 — сходимость статистического математического ожидания M*(n) выбранной
        характеристики выбранного закона (ДИ при заданной β);
    5 — сходимость исправленной выборочной дисперсии D̃(n);
    6 — точность оценивания ε(β,n) с линией заданной точности ε₀ и оценкой
        потребного объёма испытаний.

    Возвращает имя графика → список путей (по одному на формат).
    """
    import matplotlib
    from pathlib import Path

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {}
    models = study["models"]
    colors = plt.cm.tab10.colors
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    def save(fig, name: str) -> None:
        fig.tight_layout()
        paths[name] = []
        for fmt in formats:
            p = out / f"{name}.{fmt}"
            fig.savefig(p, dpi=150)
            paths[name].append(p)
        plt.close(fig)

    # --- График 1: реализации четырёх законов ---------------------------------
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.6))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        res = examples[model]
        dx, dy = res.district.exterior.xy
        ax.plot(dx, dy, "k-", linewidth=1.2)
        for chord in res.line_chords:
            cx, cy = chord.xy
            ax.plot(cx, cy, color="0.5", linestyle="--", linewidth=1.0)
        for c in res.cluster_centers:
            ax.plot(c.x, c.y, marker="x", color="red", markersize=9, markeredgewidth=2.0)
        for o in res.objects:
            x, y = o.geometry.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True,
                                    facecolor=model_colors[model], edgecolor="black",
                                    linewidth=0.5, alpha=0.75))
        ax.set_aspect("equal")
        ax.set_title(MODEL_TITLES_RU[model], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("График 1. Реализации случайного пространственного распределения "
                 "объектов наблюдения", fontsize=12)
    save(fig, "stat_chart1_realizations")

    # --- График 2: эмпирические и эталонные F(r) ------------------------------
    lam = None
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 5.2))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        emp = nn_empirical.get(model)
        if emp is None or emp.size == 0:
            continue
        r_max = float(np.percentile(emp, 99.5))
        xs, ys = _empirical_cdf(emp)
        ax.step(xs, ys, where="post", color=model_colors[model], linewidth=1.6,
                label=f"эмпирическая F*(r), {study['realizations']} реализаций")
        if nn_reference is not None and model in nn_reference and nn_reference[model].size:
            xr, yr = _empirical_cdf(nn_reference[model])
            ax.step(xr, yr, where="post", color="black", linewidth=1.1, alpha=0.8,
                    linestyle="--", label="имитационный эталон (независимые seed)")
        if lam is None:
            lam = study["per_model"][model]["runs"][0]["n_objects"] / study["district_area_m2"]
        rgrid = np.linspace(0.0, r_max, 400)
        ax.plot(rgrid, rayleigh_cdf(rgrid, lam), color="red", linewidth=1.3,
                linestyle=":", label="теоретическая F(r) для CSR (Рэлей)")
        ax.set_xlim(0.0, r_max)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(MODEL_TITLES_RU[model], fontsize=11)
        ax.set_xlabel("r, м")
        ax.set_ylabel("F(r)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("График 2. Эмпирические и эталонные функции распределения расстояний "
                 "до ближайшего соседа\n(краевая поправка граничной полосы; запрет "
                 "пересечений учитывается имитационным эталоном)", fontsize=12)
    save(fig, "stat_chart2_nn_cdf")

    # --- График 3: сравнение законов по индексам ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metr, title in (
        (axes[0], "ce_r", "Индекс Кларка–Эванса R"),
        (axes[1], "dispersion_index", "Индекс квадрантной дисперсии I"),
        (axes[2], "nn_bearing_r", "Характеристика направленности R_b"),
    ):
        means, halves, labels, cols = [], [], [], []
        for model in models:
            last = study["per_model"][model]["classical"][metr][-1]
            means.append(last["m_star"])
            halves.append(last["epsilon"])
            labels.append(MODEL_TITLES_RU[model])
            cols.append(model_colors[model])
        x = np.arange(len(models))
        ax.bar(x, means, yerr=halves, capsize=5, color=cols, alpha=0.8,
               edgecolor="black", linewidth=0.5)
        ax.axhline(1.0 if metr != "nn_bearing_r" else 0.0,
                   color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=12)
        ax.set_title(f"{title}\n(M* ± ε, β = {beta:.2f}, n = {study['realizations']})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("График 3. Статистические характеристики пространственного распределения",
                 fontsize=12)
    save(fig, "stat_chart3_indices")

    # --- Графики 4–6: сходимость оценок выбранной характеристики --------------
    if selected_model not in study["per_model"]:
        raise ValueError(f"render_statistics_charts: закон «{selected_model}» отсутствует в исследовании.")
    if metric not in study["per_model"][selected_model]["classical"]:
        raise ValueError(f"render_statistics_charts: неизвестная характеристика «{metric}».")
    series = study["per_model"][selected_model]["classical"][metric]
    ns = np.array([r["n"] for r in series], dtype=float)
    m_star = np.array([r["m_star"] for r in series], dtype=float)
    d_tilde = np.array([r["d_tilde"] for r in series], dtype=float)
    eps = np.array([r["epsilon"] for r in series], dtype=float)
    lo = np.array([r["ci_low"] for r in series], dtype=float)
    hi = np.array([r["ci_high"] for r in series], dtype=float)
    title_ru = study["metric_titles_ru"].get(metric, metric)
    law_ru = MODEL_TITLES_RU[selected_model]
    color = model_colors[selected_model]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(ns, lo, hi, color=color, alpha=0.25, label=f"ДИ I(β,n), β = {beta:.2f}")
    ax.plot(ns, m_star, color=color, linewidth=1.8, label="M*(n)")
    ax.axhline(m_star[-1], color="gray", linestyle=":", linewidth=1.0,
               label=f"M*({int(ns[-1])}) = {m_star[-1]:.4f}")
    ax.set_title(f"График 4. Статистическое математическое ожидание M*(n)\n"
                 f"{title_ru} — {law_ru}", fontsize=11)
    ax.set_xlabel("объём испытаний n (независимых реализаций)")
    ax.set_ylabel("M*(n)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, "stat_chart4_mean")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ns, d_tilde, color=color, linewidth=1.8, label="D̃(n) (деление на n−1)")
    ax.axhline(d_tilde[-1], color="gray", linestyle=":", linewidth=1.0,
               label=f"D̃({int(ns[-1])}) = {d_tilde[-1]:.5g}")
    ax.set_title(f"График 5. Исправленная выборочная дисперсия D̃(n)\n"
                 f"{title_ru} — {law_ru}", fontsize=11)
    ax.set_xlabel("объём испытаний n (независимых реализаций)")
    ax.set_ylabel("D̃(n)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, "stat_chart5_variance")

    n_req = required_trials(float(series[-1]["sigma_tilde"]), epsilon0, beta=beta)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ns, eps, color=color, linewidth=1.8, label="ε(β,n) = t·σ̃/√n")
    ax.axhline(epsilon0, color="green", linestyle="--", linewidth=1.3,
               label=f"заданная точность ε₀ = {epsilon0:.4g}")
    if n_req["n_required"] <= ns[-1]:
        ax.axvline(n_req["n_required"], color="red", linestyle=":", linewidth=1.2,
                   label=f"потребный объём n ≥ {n_req['n_required']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"График 6. Точность оценивания ε(β,n), β = {beta:.2f}\n"
                 f"{title_ru} — {law_ru}", fontsize=11)
    ax.set_xlabel("объём испытаний n (лог. шкала); потребный объём — пилотная оценка n ≥ (z·σ̃/ε₀)²")
    ax.set_ylabel("ε(β,n) (лог. шкала)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, "stat_chart6_precision")

    return paths
