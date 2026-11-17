"""Оркестрация генерации сценария: район -> объекты -> кадры."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from .config import ScenarioConfig
from .frames import Frame, build_frames
from .geometry import LocalCRS, build_district_polygon
from .placement import PlacementReport, PlacedObject, place_objects


@dataclass
class Scenario:
    """Результат генерации: район, объекты, кадры и отчёт."""

    config: ScenarioConfig
    local: LocalCRS
    district_local: Polygon
    objects: list[PlacedObject]
    frames: list[Frame]
    report: PlacementReport
    warnings: list[str] = field(default_factory=list)


def generate_scenario(config: ScenarioConfig) -> Scenario:
    """Сгенерировать сценарий целиком. Одинаковые config и seed дают одинаковый результат."""
    rng = np.random.default_rng(config.seed)
    local = LocalCRS(config.district.center_lat, config.district.center_lon)
    width_m = config.district.width_km * 1000.0
    height_m = config.district.height_km * 1000.0
    district = build_district_polygon(local, width_m, height_m, config.district.rotation_deg)

    manual = [(c.object_id, c.latitude, c.longitude) for c in config.placement.manual_centers]
    objects, report = place_objects(
        rng, district, config.placement, config.objects, config.attributes, manual, local.to_xy
    )
    frames = build_frames(config.frames, objects)

    warnings = list(report.warnings)
    if manual:
        warnings.append(f"Ручных центров: {len(manual)} из {config.objects.count} объектов.")
    if frames:
        low = [f for f in frames if f.coverage_fraction < 1.0]
        if low:
            warnings.append(
                f"{len(low)} кадров не полностью покрывают объект (coverage_fraction < 1)."
            )
    return Scenario(config, local, district, objects, frames, report, warnings)
