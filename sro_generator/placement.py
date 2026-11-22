"""Случайное размещение объектов внутри района.

Семантика ограничений (единая для версии 1.1.0), функция :func:`pair_placement_ok`:

- полигоны **не пересекаются** (в том числе касаются границ без пересечения
  по площади): расстояние между ними должно быть не меньше ``min_distance_m``;
- полигоны пересекаются **по положительной площади**
  (``area(∩) > 1e-12 · min(area)``):
    - ``allow_overlap = false`` — размещение запрещено;
    - ``allow_overlap = true`` — допускается, если
      ``area(∩) / min(area) ≤ max_overlap_fraction``.

Касание границ (площадь пересечения ≈ 0, расстояние 0) трактуется как
«не пересекаются»: при ``min_distance_m = 0`` касание допустимо,
при ``min_distance_m > 0`` — отклоняется правилом расстояния.

Допустимая область центров объекта вычисляется точно — как пересечение
полуплоскостей (эрозия Минковского выпуклой области размещения на опорную
функцию объекта), а не через описанный радиус. Режим ``legacy`` сохраняет
поведение 1.0.0 (``inner.buffer(-R)``). Поиск конфликтов ускоряется
пространственным индексом ``shapely.STRtree`` с периодическим перестроением;
решения (принять/отклонить) полностью совпадают с полным перебором.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .config import AttributesConfig, ObjectsConfig, PlacementConfig
from .geometry import generate_polygon_shape, is_proper_polygon

QUADRANT_GRID = 3  # район делится на 3x3 ячейки (должен совпадать со stats.QUADRANT_GRID)

_STRTREE_REBUILD_THRESHOLD = 64  # перестроение индекса после стольких новых объектов
_POSITIVE_AREA_REL_TOL = 1e-12   # доля площади, ниже которой пересечение считается касанием
_OVERLAP_ABS_TOL = 1e-12         # абсолютный допуск доли перекрытия
_DISTANCE_ABS_TOL = 1e-9         # абсолютный допуск расстояния, м
_HUGE = 1e9                      # характерный масштаб «бесконечности» для полуплоскостей


class PlacementError(RuntimeError):
    """Контролируемая ошибка размещения (объект невозможно разместить)."""


@dataclass
class PlacedObject:
    """Размещённый объект с геометрией (локальные метры) и атрибутами."""

    object_id: str
    geometry: Polygon
    center: Point
    area_m2: float
    vertex_count: int
    rotation_deg: float
    class_label: str | None
    priority: int
    deadline_h: float
    resolution_m: float
    placement_mode: str
    coordinate_fixed: bool
    attempts: int = 1


@dataclass
class PlacementReport:
    """Итоги размещения."""

    attempts_total: int = 0
    warnings: list[str] = field(default_factory=list)
    exact_checks_total: int = 0      # точных геометрических проверок пар
    tree_queries: int = 0            # запросов к STRtree
    tree_rebuilds: int = 0           # перестроений STRtree
    center_region_area_sum_m2: float = 0.0   # сумма площадей допустимых областей случайных объектов
    center_region_area_count: int = 0        # число случайных объектов с посчитанной областью
    grid_cell_area_fraction_sums: list[float] = field(default_factory=lambda: [0.0] * (QUADRANT_GRID * QUADRANT_GRID))
    grid_cell_area_fraction_count: int = 0

    @property
    def center_region_area_mean_m2(self) -> float:
        if self.center_region_area_count == 0:
            return 0.0
        return self.center_region_area_sum_m2 / self.center_region_area_count

    def grid_cell_area_fractions_mean(self) -> list[float]:
        """Средние доли площади допустимой области в ячейках сетки 3×3 (по случайным объектам)."""
        if self.grid_cell_area_fraction_count == 0:
            return [0.0] * (QUADRANT_GRID * QUADRANT_GRID)
        return [s / self.grid_cell_area_fraction_count for s in self.grid_cell_area_fraction_sums]


def pair_placement_ok(
    a: Polygon,
    b: Polygon,
    min_distance_m: float,
    allow_overlap: bool,
    max_overlap_fraction: float,
) -> bool:
    """Симметричная проверка пары полигонов на совместимость по расстоянию/перекрытию.

    См. модульную семантику в docstring модуля. Проверка симметрична
    (не зависит от порядка аргументов) и устойчива к численному шуму:
    пересечение площадью меньше ``1e-12 · min(area)`` трактуется как касание.
    """
    inter_area = a.intersection(b).area
    min_area = min(a.area, b.area)
    if inter_area > _POSITIVE_AREA_REL_TOL * max(min_area, 1e-300):
        if not allow_overlap:
            return False
        return (inter_area / min_area) <= max_overlap_fraction + _OVERLAP_ABS_TOL
    return a.distance(b) >= min_distance_m - _DISTANCE_ABS_TOL


def _circumradius(poly: Polygon, cx: float, cy: float) -> float:
    """Максимальное расстояние от центра до вершин полигона."""
    return max(math.hypot(x - cx, y - cy) for x, y in poly.exterior.coords[:-1])


def _is_convex_polygon(poly: Polygon, rel_tol: float = 1e-9) -> bool:
    """Проверка выпуклости многоугольника (относительный допуск к длинам рёбер)."""
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = coords[i - 1]
        bx, by = coords[i]
        cx, cy = coords[(i + 1) % n]
        e1x, e1y = bx - ax, by - ay
        e2x, e2y = cx - bx, cy - by
        cross = e1x * e2y - e1y * e2x
        scale = max(1.0, math.hypot(e1x, e1y) * math.hypot(e2x, e2y))
        if abs(cross) <= rel_tol * scale:
            continue  # практически коллинеарные рёбра (аппроксимация дуг) не ломают выпуклость
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def exact_center_region(inner: Polygon, shape: Polygon) -> Polygon | None:
    """Точная область допустимых центров ``{t : shape + t ⊆ inner}`` для выпуклого ``inner``.

    Для каждого ребра ``inner`` с единичной внешней нормалью ``n`` и смещением ``c``
    ограничение имеет вид ``n·t ≤ c − h``, где ``h = max(n·v)`` по опорным точкам
    ``shape``. Опорная функция линейна, поэтому максимум достигается на вершинах
    выпуклой оболочки — перебор всех вершин контура достаточен для любой формы.
    Результат — точный полигон (пересечение полуплоскостей), без потерь площади
    от описанного радиуса.

    Возвращает ``None``, если ``inner`` невыпукл, область пуста или вырождена —
    вызывающий код должен использовать запасной вариант.
    """
    if not isinstance(inner, Polygon) or inner.is_empty or inner.area <= 0.0:
        return None
    if not _is_convex_polygon(inner):
        return None
    shape_pts = list(shape.exterior.coords)[:-1]
    if not shape_pts:
        return None
    coords = list(inner.exterior.coords)[:-1]
    # нормаль (dy, −dx) внешняя только для обхода против часовой стрелки;
    # ориентация кольца от shapely (в т.ч. после buffer) не гарантирована — нормализуем
    def _signed_area(ring: list[tuple[float, float]]) -> float:
        return 0.5 * sum(
            ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
            for i in range(len(ring))
        )
    if _signed_area(coords) < 0.0:
        coords = coords[::-1]
    region: BaseGeometry | None = None
    n = len(coords)
    for i in range(n):
        ax, ay = coords[i]
        bx, by = coords[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        edge_len = math.hypot(dx, dy)
        if edge_len <= 0.0:
            continue
        nx, ny = dy / edge_len, -dx / edge_len  # внешняя нормаль (обход против часовой стрелки)
        c = nx * ax + ny * ay
        h = max(nx * x + ny * y for x, y in shape_pts)
        bound = c - h
        px, py = -ny, nx  # направление вдоль граничной прямой
        halfplane = Polygon([
            (bound * nx + _HUGE * px, bound * ny + _HUGE * py),
            (bound * nx - _HUGE * px, bound * ny - _HUGE * py),
            (bound * nx - _HUGE * px - _HUGE * nx, bound * ny - _HUGE * py - _HUGE * ny),
            (bound * nx + _HUGE * px - _HUGE * nx, bound * ny + _HUGE * py - _HUGE * ny),
        ])
        region = halfplane if region is None else region.intersection(halfplane)
        if region.is_empty:
            return None
    if not isinstance(region, Polygon) or region.area <= 0.0:
        return None
    return region


def _legacy_center_region(inner: BaseGeometry, shape: Polygon) -> BaseGeometry:
    """Допустимая область центров как в 1.0.0: эрозия на описанный радиус (консервативно)."""
    rad = _circumradius(shape, 0.0, 0.0)
    return inner.buffer(-rad)


def _center_region(
    inner: BaseGeometry,
    shape: Polygon,
    mode: str,
    report: PlacementReport,
    warned: list[bool],
) -> BaseGeometry:
    """Выбор допустимой области центров: точная для выпуклой области, иначе legacy."""
    if mode == "exact" and isinstance(inner, Polygon):
        region = exact_center_region(inner, shape)
        if region is not None:
            return region
        if not warned[0]:
            report.warnings.append(
                "Область размещения невыпукла: точная эрозия недоступна, "
                "используется приближение через описанный радиус (режим legacy)."
            )
            warned[0] = True
    return _legacy_center_region(inner, shape)


def _sample_point_in_polygon(rng: np.random.Generator, poly: Polygon) -> Point:
    """Равномерная точка внутри (выпуклого) полигона: веерная триангуляция от центроида."""
    coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
    c = np.array([poly.centroid.x, poly.centroid.y], dtype=float)
    n = len(coords)
    areas = np.empty(n, dtype=float)
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        areas[i] = 0.5 * abs((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))
    total = float(areas.sum())
    pick = float(rng.random()) * total
    acc = 0.0
    idx = n - 1
    for i in range(n):
        acc += areas[i]
        if pick <= acc:
            idx = i
            break
    a = coords[idx]
    b = coords[(idx + 1) % n]
    u = float(rng.random())
    v = float(rng.random())
    if u + v > 1.0:  # отражение в треугольник
        u = 1.0 - u
        v = 1.0 - v
    p = c + u * (a - c) + v * (b - c)
    return Point(float(p[0]), float(p[1]))


class _PlacementIndex:
    """Индекс размещённых объектов: STRtree с периодическим перестроением.

    STRtree в Shapely 2.x неизменяем — индекс строится по префиксу списка
    ``placed``; объекты, добавленные после последнего перестроения, проверяются
    отдельно линейным проходом (их немного, ≤ порога перестроения). Запрос —
    ограничивающий прямоугольник кандидата, расширенный на ``min_distance_m``:
    он гарантированно пересекается с ограничивающим прямоугольником любого
    потенциально конфликтующего объекта (пересечение по площади ⇒ пересечение
    прямоугольников; расстояние < min ⇒ расстояние прямоугольников < min).
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._tree: STRtree | None = None
        self._built_upto = 0

    def notify_added(self, placed: list[PlacedObject], report: PlacementReport) -> None:
        if not self.enabled:
            return
        if self._tree is None or (len(placed) - self._built_upto) >= _STRTREE_REBUILD_THRESHOLD:
            self._tree = STRtree([p.geometry for p in placed])
            self._built_upto = len(placed)
            report.tree_rebuilds += 1

    def candidates(self, placed: list[PlacedObject], qbox: BaseGeometry, report: PlacementReport) -> list[int]:
        """Индексы объектов, которые могут нарушить ограничения относительно запроса qbox."""
        if not self.enabled or self._tree is None:
            return list(range(len(placed)))
        report.tree_queries += 1
        found = [int(i) for i in self._tree.query(qbox)]
        # объекты, добавленные после последнего перестроения, — точной проверкой
        for i in range(self._built_upto, len(placed)):
            if placed[i].geometry.envelope.intersects(qbox):
                found.append(i)
        return found


def _quadrant_fractions(district: Polygon, region: BaseGeometry) -> list[float]:
    """Доли площади допустимой области в ячейках сетки 3×3 над границами района."""
    minx, miny, maxx, maxy = district.bounds
    sx = (maxx - minx) / QUADRANT_GRID
    sy = (maxy - miny) / QUADRANT_GRID
    total = float(region.area)
    if total <= 0.0:
        return [0.0] * (QUADRANT_GRID * QUADRANT_GRID)
    fractions: list[float] = []
    for iy in range(QUADRANT_GRID):
        for ix in range(QUADRANT_GRID):
            cell = box(minx + ix * sx, miny + iy * sy,
                       minx + (ix + 1) * sx, miny + (iy + 1) * sy)
            fractions.append(float(region.intersection(cell).area) / total)
    return fractions


def place_objects(
    rng: np.random.Generator,
    district: Polygon,
    placement: PlacementConfig,
    objects_cfg: ObjectsConfig,
    attributes_cfg: AttributesConfig,
    manual_centers: list[tuple[str, float, float]],
    local_to_xy,
    *,
    use_spatial_index: bool | None = None,
    center_region_mode: str | None = None,
) -> tuple[list[PlacedObject], PlacementReport]:
    """Разместить ровно objects_cfg.count объектов: сначала ручные центры, затем случайные.

    При невозможности размещения выбрасывает PlacementError с object_id и советом,
    а не возвращает неполный набор. Параметры ``use_spatial_index`` и
    ``center_region_mode`` переопределяют значения из конфигурации (для
    сравнительных экспериментов и тестов); по умолчанию берутся из конфигурации.
    Результаты (принято/отклонено) при одинаковом наборе геометрий не зависят
    от режима индекса: STRtree отсекает только заведомо неподходящих кандидатов.
    """
    report = PlacementReport()
    placed: list[PlacedObject] = []
    index_enabled = placement.spatial_index if use_spatial_index is None else use_spatial_index
    region_mode = placement.center_region_mode if center_region_mode is None else center_region_mode
    index = _PlacementIndex(index_enabled)
    warned_region = [False]
    inner = district.buffer(-placement.boundary_margin_m)
    if inner.is_empty or inner.area <= 0.0:
        raise PlacementError(
            "placement.boundary_margin_m: отступ от границы слишком велик для заданного района. "
            "Уменьшите отступ или увеличьте район."
        )

    def make_shape() -> tuple[Polygon, int, float]:
        """Форма объекта вокруг начала координат + число вершин и угол поворота."""
        n = int(rng.integers(objects_cfg.vertices_min, objects_cfg.vertices_max + 1))
        target_area = float(rng.uniform(objects_cfg.area_min_m2, objects_cfg.area_max_m2))
        rot = float(rng.uniform(objects_cfg.rotation_min_deg, objects_cfg.rotation_max_deg))
        for _ in range(200):
            try:
                poly = generate_polygon_shape(
                    rng, n, target_area, objects_cfg.irregularity, rot, objects_cfg.max_aspect_ratio
                )
            except ValueError:
                continue
            if is_proper_polygon(poly, n):
                return poly, n, rot
        raise PlacementError("Не удалось построить корректную форму объекта за 200 попыток.")

    def make_attrs(mode: str) -> dict:
        return {
            "class_label": (
                str(rng.choice(attributes_cfg.class_labels)) if attributes_cfg.class_labels else None
            ),
            "priority": int(rng.integers(attributes_cfg.priority_min, attributes_cfg.priority_max + 1)),
            "deadline_h": float(rng.choice(attributes_cfg.deadline_choices_h)),
            "resolution_m": float(rng.choice(attributes_cfg.resolution_choices_m)),
            "placement_mode": mode,
        }

    def add_object(poly: Polygon, cx: float, cy: float, n: int, rot: float, object_id: str, fixed: bool, mode: str, attempts: int) -> None:
        shifted = Polygon([(x + cx, y + cy) for x, y in poly.exterior.coords[:-1]])
        placed.append(
            PlacedObject(
                object_id=object_id,
                geometry=shifted,
                center=Point(cx, cy),
                area_m2=shifted.area,
                vertex_count=n,
                rotation_deg=rot,
                coordinate_fixed=fixed,
                **make_attrs(mode),
                attempts=attempts,
            )
        )
        index.notify_added(placed, report)

    def conflicts_with_placed(shifted: Polygon) -> bool:
        """Точная проверка кандидата против размещённых (с предварительным отсечением)."""
        minx, miny, maxx, maxy = shifted.bounds
        m = placement.min_distance_m
        qbox = box(minx - m, miny - m, maxx + m, maxy + m)
        for i in index.candidates(placed, qbox, report):
            report.exact_checks_total += 1
            if not pair_placement_ok(
                shifted, placed[i].geometry,
                placement.min_distance_m, placement.allow_overlap, placement.max_overlap_fraction,
            ):
                return True
        return False

    def check_position(shifted: Polygon, object_id: str, hint: str) -> None:
        """Проверка: полигон внутри района с отступом, расстояния и перекрытия."""
        if not inner.covers(shifted):
            raise PlacementError(
                f"Объект {object_id}: полигон выходит за границу района с учётом отступа "
                f"boundary_margin_m. {hint}"
            )
        if conflicts_with_placed(shifted):
            raise PlacementError(
                f"Объект {object_id}: нарушено минимальное расстояние min_distance_m или допустимое "
                f"перекрытие max_overlap_fraction. {hint}"
            )

    def account_region(region: BaseGeometry) -> None:
        """Учёт площади допустимой области и долей ячеек сетки (для статистики)."""
        report.center_region_area_sum_m2 += float(region.area)
        report.center_region_area_count += 1
        fr = _quadrant_fractions(district, region)
        for i, f in enumerate(fr):
            report.grid_cell_area_fraction_sums[i] += f
        report.grid_cell_area_fraction_count += 1

    # 1) объекты с фиксированными (ручными) центрами — центры не сдвигаются
    for object_id, lat, lon in manual_centers:
        cx, cy = local_to_xy(lon, lat)
        poly, n, rot = make_shape()
        shifted = Polygon([(x + cx, y + cy) for x, y in poly.exterior.coords[:-1]])
        check_position(
            shifted,
            f"{object_id} (ручной центр)",
            "Уменьшите площадь объекта или отступ, либо измените координаты центра.",
        )
        add_object(poly, cx, cy, n, rot, object_id, True, "manual", 1)

    # 2) случайные объекты
    n_random = objects_cfg.count - len(manual_centers)
    for _ in range(n_random):
        object_id = f"OBJ-{len(placed) + 1:03d}"
        poly, n, rot = make_shape()
        # форма и целевая площадь при повторных попытках сохраняются, меняется положение
        region = _center_region(inner, poly, region_mode, report, warned_region)
        hint = (
            "Уменьшите площади объектов, число объектов, отступы (boundary_margin_m, min_distance_m) "
            "или увеличьте район."
        )
        if region.is_empty or region.area <= 0.0:
            raise PlacementError(f"Объект {object_id}: не осталось допустимой области для размещения. {hint}")
        account_region(region)
        ok = False
        if region_mode == "legacy":
            # поведение 1.0.0: равномерный выбор в ограничивающем прямоугольнике с отказом
            minx, miny, maxx, maxy = region.bounds
            for attempt in range(1, placement.max_attempts_per_object + 1):
                report.attempts_total += 1
                p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if not region.covers(p):
                    continue
                shifted = Polygon([(x + p.x, y + p.y) for x, y in poly.exterior.coords[:-1]])
                if not conflicts_with_placed(shifted):
                    add_object(poly, p.x, p.y, n, rot, object_id, False, "random", attempt)
                    ok = True
                    break
        else:
            # равномерный выбор внутри точной области (триангуляция)
            for attempt in range(1, placement.max_attempts_per_object + 1):
                report.attempts_total += 1
                p = _sample_point_in_polygon(rng, region) if isinstance(region, Polygon) else None
                if p is None:
                    continue
                shifted = Polygon([(x + p.x, y + p.y) for x, y in poly.exterior.coords[:-1]])
                if not inner.covers(shifted):
                    continue  # страховка от численного выхода вершин за границу
                if not conflicts_with_placed(shifted):
                    add_object(poly, p.x, p.y, n, rot, object_id, False, "random", attempt)
                    ok = True
                    break
        if not ok:
            raise PlacementError(
                f"Объект {object_id}: не удалось разместить за {placement.max_attempts_per_object} "
                f"попыток. {hint}"
            )

    return placed, report
