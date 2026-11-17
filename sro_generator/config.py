"""Модели конфигурации и проверка входных данных.

Все сообщения об ошибках — на русском языке, с именем параметра и способом исправления.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConfigError(Exception):
    """Ошибка конфигурации с понятным русским сообщением (пробрасывается напрямую)."""


class DistrictConfig(BaseModel):
    """Параметры прямоугольного района съёмки (WGS 84)."""

    center_lat: float = Field(..., ge=-90.0, le=90.0)
    center_lon: float = Field(..., ge=-180.0, le=180.0)
    width_km: float = Field(..., gt=0.0)
    height_km: float = Field(..., gt=0.0)
    rotation_deg: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> "DistrictConfig":
        if not (-90.0 <= self.center_lat <= 90.0):
            raise ConfigError("district.center_lat: широта должна быть от -90 до 90 градусов.")
        if not (-180.0 <= self.center_lon <= 180.0):
            raise ConfigError("district.center_lon: долгота должна быть от -180 до 180 градусов.")
        if self.width_km <= 0:
            raise ConfigError("district.width_km: ширина района должна быть положительной (например, 100.0).")
        if self.height_km <= 0:
            raise ConfigError("district.height_km: высота района должна быть положительной (например, 90.0).")
        return self


class ObjectsConfig(BaseModel):
    """Параметры генерируемых объектов."""

    count: int = Field(..., ge=1)
    vertices_min: int = Field(..., ge=3, le=10)
    vertices_max: int = Field(..., ge=3, le=10)
    area_min_m2: float = Field(..., gt=0.0)
    area_max_m2: float = Field(..., gt=0.0)
    irregularity: float = Field(..., ge=0.0, le=1.0)
    rotation_min_deg: float = 0.0
    rotation_max_deg: float = 360.0
    max_aspect_ratio: float = Field(..., gt=1.0)

    @model_validator(mode="after")
    def _check(self) -> "ObjectsConfig":
        if self.count < 1:
            raise ConfigError("objects.count: число объектов должно быть целым не меньше 1.")
        if not (3 <= self.vertices_min <= 10):
            raise ConfigError("objects.vertices_min: число вершин должно быть в пределах 3…10.")
        if not (3 <= self.vertices_max <= 10):
            raise ConfigError("objects.vertices_max: число вершин должно быть в пределах 3…10.")
        if self.vertices_min > self.vertices_max:
            raise ConfigError("objects.vertices_min: минимум числа вершин не может превышать vertices_max.")
        if self.area_min_m2 <= 0 or self.area_max_m2 <= 0:
            raise ConfigError("objects.area_min_m2/area_max_m2: площади должны быть положительными.")
        if self.area_min_m2 > self.area_max_m2:
            raise ConfigError("objects.area_min_m2: минимальная площадь не может превышать area_max_m2.")
        if not (0.0 <= self.irregularity <= 1.0):
            raise ConfigError("objects.irregularity: нерегулярность формы должна быть от 0 до 1.")
        if self.rotation_min_deg > self.rotation_max_deg:
            raise ConfigError("objects.rotation_min_deg: минимум поворота не может превышать rotation_max_deg.")
        if self.max_aspect_ratio <= 1.0:
            raise ConfigError("objects.max_aspect_ratio: отношение сторон должно быть больше 1.")
        return self


class ManualCenter(BaseModel):
    """Фиксированный центр объекта (ручной режим)."""

    object_id: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


class PlacementConfig(BaseModel):
    """Параметры случайного размещения."""

    mode: Literal["random"] = "random"
    min_distance_m: float = Field(..., ge=0.0)
    boundary_margin_m: float = Field(..., ge=0.0)
    allow_overlap: bool = False
    max_overlap_fraction: float = Field(0.0, ge=0.0, le=1.0)
    max_attempts_per_object: int = Field(..., gt=0)
    manual_centers: list[ManualCenter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "PlacementConfig":
        if self.min_distance_m < 0:
            raise ConfigError("placement.min_distance_m: расстояние не может быть отрицательным.")
        if self.boundary_margin_m < 0:
            raise ConfigError("placement.boundary_margin_m: отступ не может быть отрицательным.")
        if not (0.0 <= self.max_overlap_fraction <= 1.0):
            raise ConfigError("placement.max_overlap_fraction: доля перекрытия должна быть от 0 до 1.")
        if not self.allow_overlap and self.max_overlap_fraction > 0.0:
            raise ConfigError(
                "placement.max_overlap_fraction: перекрытие запрещено (allow_overlap=false), задайте 0.0."
            )
        if self.max_attempts_per_object < 1:
            raise ConfigError("placement.max_attempts_per_object: число попыток должно быть положительным.")
        ids = [c.object_id for c in self.manual_centers]
        if len(ids) != len(set(ids)):
            raise ConfigError("placement.manual_centers: идентификаторы object_id ручных центров должны быть уникальными.")
        for c in self.manual_centers:
            if not (-90.0 <= c.latitude <= 90.0):
                raise ConfigError(f"placement.manual_centers[{c.object_id}]: широта должна быть от -90 до 90.")
            if not (-180.0 <= c.longitude <= 180.0):
                raise ConfigError(f"placement.manual_centers[{c.object_id}]: долгота должна быть от -180 до 180.")
        return self


class AttributesConfig(BaseModel):
    """Параметры атрибутов объектов."""

    class_labels: list[str] = Field(default_factory=list)
    priority_min: int = Field(1, ge=1)
    priority_max: int = Field(5, ge=1)
    deadline_choices_h: list[float] = Field(default_factory=lambda: [30.0, 48.0])
    resolution_choices_m: list[float] = Field(default_factory=lambda: [0.5, 1.0])

    @model_validator(mode="after")
    def _check(self) -> "AttributesConfig":
        if self.priority_min > self.priority_max:
            raise ConfigError("attributes.priority_min: минимум приоритета не может превышать priority_max.")
        if not self.deadline_choices_h:
            raise ConfigError("attributes.deadline_choices_h: список директивных сроков не может быть пустым.")
        if not self.resolution_choices_m:
            raise ConfigError("attributes.resolution_choices_m: список разрешений не может быть пустым.")
        if any(d <= 0 for d in self.deadline_choices_h):
            raise ConfigError("attributes.deadline_choices_h: сроки должны быть положительными.")
        if any(r <= 0 for r in self.resolution_choices_m):
            raise ConfigError("attributes.resolution_choices_m: разрешения должны быть положительными.")
        return self


class FramesConfig(BaseModel):
    """Параметры простых кадров-кандидатов РСА."""

    enabled: bool = False
    along_track_km: float = Field(5.0, gt=0.0)
    cross_track_km: float = Field(5.0, gt=0.0)
    pass_azimuths_deg: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "FramesConfig":
        if self.along_track_km <= 0 or self.cross_track_km <= 0:
            raise ConfigError("frames.along_track_km/cross_track_km: размеры кадра должны быть положительными.")
        if self.enabled and not self.pass_azimuths_deg:
            raise ConfigError("frames.pass_azimuths_deg: задайте хотя бы один азимут пролёта или отключите кадры.")
        return self


class ScenarioConfig(BaseModel):
    """Полная конфигурация сценария генерации."""

    name: str = "district"
    seed: int
    district: DistrictConfig
    objects: ObjectsConfig
    placement: PlacementConfig
    attributes: AttributesConfig = Field(default_factory=AttributesConfig)
    frames: FramesConfig = Field(default_factory=FramesConfig)

    @model_validator(mode="after")
    def _check(self) -> "ScenarioConfig":
        if len(self.placement.manual_centers) > self.objects.count:
            raise ConfigError(
                "placement.manual_centers: число ручных центров не может превышать objects.count."
            )
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "ScenarioConfig":
        """Загрузить и проверить конфигурацию из JSON-файла (UTF-8)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls.model_validate(data)
        except ConfigError:
            raise
        except Exception as exc:  # pydantic.ValidationError и прочее
            raise ConfigError(f"Ошибка в конфигурации JSON: {exc}") from exc

    def to_dict(self) -> dict:
        return self.model_dump()
