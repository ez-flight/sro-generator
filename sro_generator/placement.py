"""Случайное размещение объектов внутри района."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from .config import AttributesConfig, ObjectsConfig, PlacementConfig
from .geometry import generate_polygon_shape, is_proper_polygon


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


def _circumradius(poly: Polygon, cx: float, cy: float) -> float:
    """Максимальное расстояние от центра до вершин полигона."""
    return max(math.hypot(x - cx, y - cy) for x, y in poly.exterior.coords[:-1])


def _distance_ok(
    poly: Polygon,
    placed: list[PlacedObject],
    min_distance_m: float,
    allow_overlap: bool,
    max_overlap_fraction: float,
) -> bool:
    """Проверка минимального расстояния между границами и допустимого перекрытия."""
    for other in placed:
        if poly.distance(other.geometry) < min_distance_m - 1e-9:
            return False
        if poly.intersects(other.geometry):
            inter_area = poly.intersection(other.geometry).area
            if inter_area > 0.0:
                if not allow_overlap:
                    return False
                if inter_area / min(poly.area, other.geometry.area) > max_overlap_fraction + 1e-12:
                    return False
    return True


def place_objects(
    rng: np.random.Generator,
    district: Polygon,
    placement: PlacementConfig,
    objects_cfg: ObjectsConfig,
    attributes_cfg: AttributesConfig,
    manual_centers: list[tuple[str, float, float]],
    local_to_xy,
) -> tuple[list[PlacedObject], PlacementReport]:
    """Разместить ровно objects_cfg.count объектов: сначала ручные центры, затем случайные.

    При невозможности размещения выбрасывает PlacementError с object_id и советом,
    а не возвращает неполный набор.
    """
    report = PlacementReport()
    placed: list[PlacedObject] = []
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

    def check_position(shifted: Polygon, object_id: str, hint: str) -> None:
        """Проверка: полигон внутри района с отступом, расстояния и перекрытия."""
        if not inner.covers(shifted):
            raise PlacementError(
                f"Объект {object_id}: полигон выходит за границу района с учётом отступа "
                f"boundary_margin_m. {hint}"
            )
        if not _distance_ok(shifted, placed, placement.min_distance_m, placement.allow_overlap, placement.max_overlap_fraction):
            raise PlacementError(
                f"Объект {object_id}: нарушено минимальное расстояние min_distance_m или допустимое "
                f"перекрытие max_overlap_fraction. {hint}"
            )

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
        rad = _circumradius(poly, 0.0, 0.0)
        region = inner.buffer(-rad)
        hint = (
            "Уменьшите площади объектов, число объектов, отступы (boundary_margin_m, min_distance_m) "
            "или увеличьте район."
        )
        if region.is_empty or region.area <= 0.0:
            raise PlacementError(f"Объект {object_id}: не осталось допустимой области для размещения. {hint}")
        minx, miny, maxx, maxy = region.bounds
        ok = False
        for attempt in range(1, placement.max_attempts_per_object + 1):
            report.attempts_total += 1
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if not region.covers(p):
                continue
            shifted = Polygon([(x + p.x, y + p.y) for x, y in poly.exterior.coords[:-1]])
            if not _distance_ok(shifted, placed, placement.min_distance_m, placement.allow_overlap, placement.max_overlap_fraction):
                continue
            add_object(poly, p.x, p.y, n, rot, object_id, False, "random", attempt)
            ok = True
            break
        if not ok:
            raise PlacementError(
                f"Объект {object_id}: не удалось разместить за {placement.max_attempts_per_object} "
                f"попыток. {hint}"
            )

    return placed, report
