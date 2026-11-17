"""Простые кадры-кандидаты РСА вокруг объектов."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.affinity import rotate
from shapely.geometry import Polygon

from .config import FramesConfig
from .placement import PlacedObject


@dataclass
class Frame:
    """Прямоугольный кадр-кандидат вокруг объекта (локальные метры)."""

    frame_id: str
    object_id: str
    pass_azimuth_deg: float
    geometry: Polygon
    center_x_m: float
    center_y_m: float
    coverage_fraction: float


def build_frames(frames_cfg: FramesConfig, objects: list[PlacedObject]) -> list[Frame]:
    """Построить по одному кадру на объект и азимут пролёта (центр — в центроиде объекта)."""
    frames: list[Frame] = []
    if not frames_cfg.enabled:
        return frames
    along = frames_cfg.along_track_km * 1000.0
    cross = frames_cfg.cross_track_km * 1000.0
    ha, hc = along / 2.0, cross / 2.0
    base = Polygon([(-hc, -ha), (hc, -ha), (hc, ha), (-hc, ha)])
    for obj in objects:
        cx, cy = obj.geometry.centroid.x, obj.geometry.centroid.y
        moved = Polygon([(x + cx, y + cy) for x, y in base.exterior.coords[:-1]])
        for az in frames_cfg.pass_azimuths_deg:
            # азимут по часовой стрелке от севера: поворот прямоугольника вокруг центра
            geom = rotate(moved, az, origin=(cx, cy), use_radians=False)
            inter = obj.geometry.intersection(geom).area
            coverage = inter / obj.geometry.area if obj.geometry.area > 0 else 0.0
            coverage = min(max(coverage, 0.0), 1.0)
            frames.append(
                Frame(
                    frame_id=f"FRM-{len(frames) + 1:04d}",
                    object_id=obj.object_id,
                    pass_azimuth_deg=float(az),
                    geometry=geom,
                    center_x_m=cx,
                    center_y_m=cy,
                    coverage_fraction=coverage,
                )
            )
    return frames
