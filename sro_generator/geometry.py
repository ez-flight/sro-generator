"""Геометрические расчёты: локальная метрическая СК, форма объектов, район.

Все расчёты площади, расстояний и пересечений выполняются в локальной
азимутальной равнопромежуточной проекции (метры), а не в градусах.
"""

from __future__ import annotations

import math

import numpy as np
from pyproj import CRS, Transformer
from shapely.affinity import rotate
from shapely.geometry import Polygon


class LocalCRS:
    """Локальная метрическая CRS (azimuthal equidistant) с центром в районе."""

    def __init__(self, center_lat: float, center_lon: float) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
        )
        self.wgs84 = CRS.from_epsg(4326)
        self.to_local = Transformer.from_crs(self.wgs84, self.crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.crs, self.wgs84, always_xy=True)

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        """WGS 84 (lon, lat) -> локальные метры (x, y)."""
        x, y = self.to_local.transform(lon, lat)
        return float(x), float(y)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Локальные метры (x, y) -> WGS 84 (lon, lat)."""
        lon, lat = self.to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def polygon_to_lonlat(self, poly: Polygon) -> Polygon:
        """Преобразовать полигон из локальных метров в WGS 84 (lon, lat)."""
        coords = [self.to_lonlat(x, y) for x, y in poly.exterior.coords]
        return Polygon(coords)


def build_district_polygon(local: LocalCRS, width_m: float, height_m: float, rotation_deg: float) -> Polygon:
    """Прямоугольник района в локальных метрах, повёрнутый на rotation_deg по часовой стрелке от севера."""
    hw, hh = width_m / 2.0, height_m / 2.0
    poly = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    return rotate(poly, rotation_deg, origin=(0.0, 0.0), use_radians=False)


def _sorted_random_angles(rng: np.random.Generator, n: int, irregularity: float) -> np.ndarray:
    """Отсортированные случайные углы с умеренным разбросом (растёт с irregularity)."""
    base = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    jitter = rng.uniform(-math.pi / n, math.pi / n, size=n) * irregularity
    angles = np.sort(base + jitter)
    # гарантируем строгий обход против часовой стрелки
    angles = np.mod(angles, 2.0 * math.pi)
    return np.sort(angles)


def polygon_aspect_ratio(poly: Polygon) -> float:
    """Отношение длинной стороны ориентированного описанного прямоугольника к короткой (≥ 1)."""
    rect = poly.minimum_rotated_rectangle
    rcoords = list(rect.exterior.coords)[:-1]
    if len(rcoords) < 3:
        return 1.0
    sides = [math.hypot(rcoords[(i + 1) % len(rcoords)][0] - rcoords[i][0],
                        rcoords[(i + 1) % len(rcoords)][1] - rcoords[i][1])
             for i in range(len(rcoords))]
    long_s = max(sides)
    short_s = max(min(sides), 1e-12)
    return long_s / short_s


def _squeeze_along_long_axis(pts: np.ndarray, ratio: float, max_aspect_ratio: float) -> np.ndarray:
    """Сжать точки вдоль длинной оси ориентированного прямоугольника до отношения сторон max_aspect_ratio."""
    poly = Polygon(pts)
    rect = poly.minimum_rotated_rectangle
    rcoords = list(rect.exterior.coords)[:-1]
    side_a = math.hypot(rcoords[1][0] - rcoords[0][0], rcoords[1][1] - rcoords[0][1])
    side_b = math.hypot(rcoords[2][0] - rcoords[1][0], rcoords[2][1] - rcoords[1][1])
    if side_a >= side_b:
        axis_vec = np.array([rcoords[1][0] - rcoords[0][0], rcoords[1][1] - rcoords[0][1]])
    else:
        axis_vec = np.array([rcoords[2][0] - rcoords[1][0], rcoords[2][1] - rcoords[1][1]])
    norm = np.linalg.norm(axis_vec)
    if norm <= 0:
        return pts
    axis_vec = axis_vec / norm
    proj = pts @ axis_vec
    center_proj = float(proj.mean())
    factor = max_aspect_ratio / ratio
    proj_new = center_proj + (proj - center_proj) * factor
    perp = np.array([-axis_vec[1], axis_vec[0]])
    proj_perp = pts @ perp
    return np.outer(proj_new, axis_vec) + np.outer(proj_perp, perp)


def generate_polygon_shape(
    rng: np.random.Generator,
    vertex_count: int,
    target_area_m2: float,
    irregularity: float,
    rotation_deg: float,
    max_aspect_ratio: float,
) -> Polygon:
    """Сгенерировать простой многоугольник заданной площади вокруг локального начала координат.

    irregularity=0 — правильный многоугольник; больше 0 — случайные радиусы и разброс углов.
    Форма масштабируется относительно центра до целевой площади. После всех преобразований
    выполняется итоговая проверка инвариантов: простота, число вершин, площадь, отношение сторон.
    """
    if vertex_count < 3:
        raise ValueError("Число вершин многоугольника должно быть не меньше 3.")
    if target_area_m2 <= 0.0:
        raise ValueError("Целевая площадь многоугольника должна быть положительной.")
    if max_aspect_ratio <= 1.0:
        raise ValueError("max_aspect_ratio должно быть больше 1.")
    angles = _sorted_random_angles(rng, vertex_count, irregularity)
    if irregularity == 0.0:
        radii = np.full(vertex_count, 1.0)
    else:
        radii = rng.uniform(1.0 - 0.6 * irregularity, 1.0 + 0.6 * irregularity, size=vertex_count)
        radii = np.clip(radii, 0.05, None)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0.0 or not poly.is_simple:
        raise ValueError("Сгенерированный многоугольник невалиден.")

    # ограничение отношения сторон через ориентированный описанный прямоугольник
    ratio = polygon_aspect_ratio(poly)
    if ratio > max_aspect_ratio:
        pts = _squeeze_along_long_axis(pts, ratio, max_aspect_ratio)
        poly = Polygon(pts)
        if not poly.is_valid or poly.area <= 0.0 or not poly.is_simple:
            raise ValueError("Многоугольник невалиден после ограничения отношения сторон.")

    # поворот контура
    if rotation_deg != 0.0:
        poly = rotate(poly, rotation_deg, origin=(0.0, 0.0), use_radians=False)

    # масштабирование до целевой площади относительно центроида
    factor = math.sqrt(target_area_m2 / poly.area)
    cx, cy = poly.centroid.x, poly.centroid.y
    pts = np.array(poly.exterior.coords)[:-1]
    pts_scaled = (pts - np.array([cx, cy])) * factor + np.array([cx, cy])
    poly = Polygon(pts_scaled)

    # Итоговая проверка всех геометрических инвариантов. Поворот и изотропное
    # масштабирование формально сохраняют отношение сторон, но из-за дискретизации
    # ориентированного прямоугольника возможны малые отклонения — при превышении
    # допуска выполняется одна корректирующая проекция, иначе форма отклоняется.
    _AREA_REL_TOL = 1e-9
    _RATIO_ABS_TOL = 1e-6
    if not poly.is_valid or poly.area <= 0.0 or not poly.is_simple:
        raise ValueError("Многоугольник невалиден после масштабирования.")
    if abs(poly.area - target_area_m2) > _AREA_REL_TOL * target_area_m2:
        raise ValueError("Площадь многоугольника не соответствует целевой после масштабирования.")
    if len(poly.exterior.coords) - 1 != vertex_count:
        raise ValueError("Число вершин изменилось при построении формы.")
    final_ratio = polygon_aspect_ratio(poly)
    if final_ratio > max_aspect_ratio + _RATIO_ABS_TOL:
        pts = np.array(poly.exterior.coords)[:-1]
        pts = _squeeze_along_long_axis(pts, final_ratio, max_aspect_ratio)
        poly = Polygon(pts)
        if not poly.is_valid or poly.area <= 0.0 or not poly.is_simple:
            raise ValueError("Многоугольник невалиден после коррекции отношения сторон.")
        if len(poly.exterior.coords) - 1 != vertex_count:
            raise ValueError("Число вершин изменилось при коррекции отношения сторон.")
        # корректировка меняет площадь и отношение сторон — повторная нормировка и контроль
        factor = math.sqrt(target_area_m2 / poly.area)
        cx, cy = poly.centroid.x, poly.centroid.y
        pts = np.array(poly.exterior.coords)[:-1]
        poly = Polygon((pts - np.array([cx, cy])) * factor + np.array([cx, cy]))
        if not poly.is_valid or poly.area <= 0.0 or not poly.is_simple:
            raise ValueError("Многоугольник невалиден после повторной нормировки.")
        final_ratio = polygon_aspect_ratio(poly)
        if final_ratio > max_aspect_ratio + _RATIO_ABS_TOL:
            raise ValueError(
                f"Не удалось уложить отношение сторон {final_ratio:.6g} в предел {max_aspect_ratio}."
            )
        if abs(poly.area - target_area_m2) > _AREA_REL_TOL * target_area_m2:
            raise ValueError("Площадь многоугольника не соответствует целевой после коррекции.")
        if len(poly.exterior.coords) - 1 != vertex_count:
            raise ValueError("Число вершин изменилось при финальной коррекции формы.")
    return poly


def is_proper_polygon(poly: Polygon, vertex_count: int) -> bool:
    """Проверка: простой полигон, ненулевая площадь, нет совпадающих соседних вершин."""
    if not isinstance(poly, Polygon) or not poly.is_valid or not poly.is_simple:
        return False
    if poly.area <= 0.0:
        return False
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) != vertex_count:
        return False
    for a, b in zip(coords, coords[1:] + coords[:1]):
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9:
            return False
    return True
