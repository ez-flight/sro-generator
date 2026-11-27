"""Совместная стохастическая модель района ДЗЗ: целевые объекты и препятствия
оптической съёмке.

Объединяет две ранее разработанные модели:

- целевые объекты наблюдения T — четыре закона пространственного размещения
  (`target_model.generate_target_objects`: равномерно-случайный, кластерный,
  регулярный, линейный);
- препятствия оптической съёмке B — поля облачности как площадные полигоны,
  порождаемые тем же ядром пространственных моделей с собственным законом
  размещения и независимым seed (объектная бинарная маска облачности версии
  2.0 из `monte_carlo.py` сохранена в своих вкладках без изменений).

Обе модели генерируются в единой локальной метрической СК (`geometry.LocalCRS`)
на одном районе; seed объектов и препятствий независимы (разные соли
SeedSequence), поэтому изменение одного seed не меняет другую модель.

Геометрия совместной модели (S — площадь, объединения по всем полигонам):

- область досъёмки РСА:      U = T ∩ B  (объединение),  S(U) = S(T ∩ B);
- видимая оптике часть:       V = T \\ B;
- исходная полнота оптического покрытия:  P_опт = S(T \\ B) / S(T);
- доля площади объектов, требующая досъёмки:  P_скр = S(T ∩ B) / S(T);
- контрольный инвариант:  S(U) + S(V) = S(T),  P_опт + P_скр = 1
  (U и V не пересекаются по построению).

Классификация объектов по доле скрытой площади f = S(объект ∩ B) / S(объект):
полностью видимый (f ≤ 1e-9), полностью скрытый (f ≥ 1 − 1e-6), иначе —
частично скрытый. Классификация и площади определяются точными геометрическими
операциями GEOS (shapely), допуски — только на численный шум.

Экспорт: GeoJSON (RFC 7946, WGS 84) слоёв objects/obstacles/reshoot/visible
и JSON-сводка — пригодны для дальнейшего использования в `plane_sar`
(планирование РСА-досъёмки по геометрии участков U).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import __version__
from .placement import PlacementError
from .target_model import (
    MODELS,
    SpatialModelParams,
    TargetModelResult,
    TargetObject,
    generate_target_objects,
)

__all__ = [
    "STATUS_VISIBLE",
    "STATUS_PARTIAL",
    "STATUS_HIDDEN",
    "JointGeometry",
    "JointModelResult",
    "compute_joint_geometry",
    "generate_joint_model",
    "joint_summary",
    "export_joint_geojson",
    "render_joint_map",
]

STATUS_VISIBLE = "visible"     # полностью видим оптике
STATUS_PARTIAL = "partial"     # частично скрыт
STATUS_HIDDEN = "hidden"       # полностью скрыт

_VISIBLE_TOL = 1e-9    # доля скрытой площади ниже — объект полностью видим
_HIDDEN_TOL = 1e-6     # доля скрытой площади выше — объект полностью скрыт
_AREA_REL_TOL = 1e-9   # относительный допуск инварианта сохранения площадей


@dataclass
class JointGeometry:
    """Геометрия совместной модели: объединения, участки досъёмки и метрики."""

    target_union: BaseGeometry
    obstacle_union: BaseGeometry
    reshoot_union: BaseGeometry      # U = T ∩ B — участки, требующие досъёмки РСА
    visible_union: BaseGeometry      # V = T \ B — видимая оптике часть объектов
    area_targets_m2: float
    area_reshoot_m2: float
    area_visible_m2: float
    p_opt: float                     # S(T \ B) / S(T)
    p_skr: float                     # S(T ∩ B) / S(T)
    per_object: list[dict]           # по каждому объекту: площади, доля, статус


def compute_joint_geometry(
    targets: list[TargetObject],
    obstacles: list[TargetObject],
) -> JointGeometry:
    """Точное геометрическое пересечение объектов T и препятствий B.

    Чистая геометрическая функция (без генерации): принимает полигоны объектов
    и препятствий, возвращает объединения, участки досъёмки U = T ∩ B, метрики
    P_опт/P_скр и пообъектную классификацию. Инвариант сохранения площадей
    S(U) + S(V) = S(T) проверяется; при нарушении — ValueError.
    """
    if not targets:
        raise ValueError("compute_joint_geometry: нужен хотя бы один объект T.")
    target_union = unary_union([o.geometry for o in targets])
    obstacle_union = (unary_union([o.geometry for o in obstacles])
                      if obstacles else Polygon())
    reshoot = target_union.intersection(obstacle_union)
    visible = target_union.difference(obstacle_union)

    s_t = float(target_union.area)
    s_u = float(reshoot.area)
    s_v = float(visible.area)
    if s_t <= 0.0:
        raise ValueError("compute_joint_geometry: суммарная площадь объектов должна быть положительной.")
    if abs((s_u + s_v) - s_t) > _AREA_REL_TOL * s_t:
        raise ValueError(
            f"compute_joint_geometry: нарушен инвариант сохранения площадей: "
            f"S(U) + S(V) = {s_u + s_v:.6f} ≠ S(T) = {s_t:.6f}."
        )

    per_object: list[dict] = []
    for o in targets:
        inter = float(o.geometry.intersection(obstacle_union).area)
        frac = min(max(inter / o.area_m2, 0.0), 1.0)
        if frac <= _VISIBLE_TOL:
            status = STATUS_VISIBLE
        elif frac >= 1.0 - _HIDDEN_TOL:
            status = STATUS_HIDDEN
            frac = 1.0
        else:
            status = STATUS_PARTIAL
        per_object.append({
            "object_id": o.object_id,
            "area_m2": float(o.area_m2),
            "hidden_area_m2": inter,
            "visible_area_m2": float(o.area_m2) - inter,
            "hidden_fraction": frac,
            "status": status,
        })

    return JointGeometry(
        target_union=target_union,
        obstacle_union=obstacle_union,
        reshoot_union=reshoot,
        visible_union=visible,
        area_targets_m2=s_t,
        area_reshoot_m2=s_u,
        area_visible_m2=s_v,
        p_opt=s_v / s_t,
        p_skr=s_u / s_t,
        per_object=per_object,
    )


@dataclass
class JointModelResult:
    """Результат генерации совместной модели района ДЗЗ."""

    district: Polygon
    targets: TargetModelResult
    obstacles: TargetModelResult | None
    geometry: JointGeometry
    target_seed: int
    obstacle_seed: int


def _seeded_rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), salt]))


_TARGET_SALT = 20261126   # совпадает с солью target_model — реализации объектов
_OBSTACLE_SALT = 20261201  # независимая соль препятствий


def generate_joint_model(
    target_params: SpatialModelParams,
    obstacle_params: SpatialModelParams | None,
    district: Polygon,
    target_seed: int,
    obstacle_seed: int,
) -> JointModelResult:
    """Сгенерировать совместную модель: объекты T и препятствия B на одном районе.

    Обе модели — в единой локальной СК (район общий), seed независимы:
    изменение obstacle_seed не меняет объекты и наоборот. obstacle_params=None
    — район без препятствий (P_скр = 0). При невозможности размещения —
    PlacementError с указанием модели.
    """
    try:
        targets = generate_target_objects(
            target_params, district, _seeded_rng(target_seed, _TARGET_SALT))
    except PlacementError as exc:
        raise PlacementError(f"joint_model (объекты): {exc}") from exc

    obstacles: TargetModelResult | None = None
    if obstacle_params is not None:
        try:
            obstacles = generate_target_objects(
                obstacle_params, district, _seeded_rng(obstacle_seed, _OBSTACLE_SALT))
        except PlacementError as exc:
            raise PlacementError(f"joint_model (препятствия): {exc}") from exc

    geometry = compute_joint_geometry(
        targets.objects, obstacles.objects if obstacles else [])
    return JointModelResult(
        district=district,
        targets=targets,
        obstacles=obstacles,
        geometry=geometry,
        target_seed=int(target_seed),
        obstacle_seed=int(obstacle_seed),
    )


def joint_summary(result: JointModelResult) -> dict:
    """Сводка совместной модели: метрики, счётчики классификации, параметры."""
    g = result.geometry
    counts = {STATUS_VISIBLE: 0, STATUS_PARTIAL: 0, STATUS_HIDDEN: 0}
    for row in g.per_object:
        counts[row["status"]] += 1
    return {
        "version": __version__,
        "mode": "joint_spatial_model",
        "target_seed": result.target_seed,
        "obstacle_seed": result.obstacle_seed,
        "target_model": result.targets.params.model,
        "obstacle_model": result.obstacles.params.model if result.obstacles else None,
        "target_params": vars(result.targets.params),
        "obstacle_params": vars(result.obstacles.params) if result.obstacles else None,
        "n_targets": len(result.targets.objects),
        "n_obstacles": len(result.obstacles.objects) if result.obstacles else 0,
        "district_area_m2": float(result.district.area),
        "area_targets_m2": g.area_targets_m2,
        "area_reshoot_m2": g.area_reshoot_m2,
        "area_visible_m2": g.area_visible_m2,
        "p_opt": g.p_opt,
        "p_skr": g.p_skr,
        "area_identity_ok": abs(g.p_opt + g.p_skr - 1.0) < 1e-9,
        "counts": counts,
        "per_object": g.per_object,
    }


def _polygonal_parts(geom: BaseGeometry) -> list[Polygon]:
    """Разложить (Multi)Polygon/GeometryCollection на полигональные части."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    parts: list[Polygon] = []
    for g in getattr(geom, "geoms", []):
        parts.extend(_polygonal_parts(g))
    return parts


def export_joint_geojson(result: JointModelResult, local) -> dict[str, dict]:
    """GeoJSON-слои совместной модели в WGS 84 (RFC 7946), совместимые с plane_sar.

    Слои: ``district`` (граница района), ``objects`` (объекты T с долей скрытой
    площади и статусом), ``obstacles`` (препятствия B), ``reshoot`` (полигоны
    участков U = T ∩ B, требующих РСА-досъёмки), ``visible`` (видимая оптике
    часть V = T \\ B). ``local`` — LocalCRS района.
    """
    from shapely.geometry import mapping

    def feature(geom, props):
        return {"type": "Feature",
                "geometry": mapping(local.polygon_to_lonlat(geom)),
                "properties": props}

    def fc(features):
        return {"type": "FeatureCollection", "features": features}

    g = result.geometry
    objects = [
        feature(o.geometry, {
            "object_id": row["object_id"],
            "model": result.targets.params.model,
            "area_m2": round(row["area_m2"], 3),
            "hidden_area_m2": round(row["hidden_area_m2"], 3),
            "hidden_fraction": round(row["hidden_fraction"], 8),
            "status": row["status"],
        })
        for o, row in zip(result.targets.objects, g.per_object)
    ]

    obstacles = [
        feature(o.geometry, {
            "obstacle_id": o.object_id,
            "model": result.obstacles.params.model if result.obstacles else None,
            "area_m2": round(o.area_m2, 3),
        })
        for o in (result.obstacles.objects if result.obstacles else [])
    ]

    reshoot = [
        feature(p, {"part": i, "area_m2": round(float(p.area), 3), "role": "reshoot_rsc"})
        for i, p in enumerate(_polygonal_parts(g.reshoot_union))
    ]
    visible = [
        feature(p, {"part": i, "area_m2": round(float(p.area), 3), "role": "visible_optical"})
        for i, p in enumerate(_polygonal_parts(g.visible_union))
    ]
    district = [feature(result.district, {
        "role": "district",
        "area_m2": round(float(result.district.area), 3),
    })]
    return {
        "district": fc(district),
        "objects": fc(objects),
        "obstacles": fc(obstacles),
        "reshoot": fc(reshoot),
        "visible": fc(visible),
    }


def render_joint_map(
    result: JointModelResult,
    path,
    layers: dict[str, bool] | None = None,
):
    """Карта совместной модели с переключаемыми слоями.

    Слои: ``district`` (граница района), ``obstacles`` (препятствия, полупрозрачные),
    ``objects`` (контуры объектов T), ``visible`` (видимая оптике часть — зелёный),
    ``reshoot`` (участки досъёмки U — красный со штриховкой). По умолчанию все
    слои включены. Возвращает путь к PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    on = {"district": True, "obstacles": True, "objects": True,
          "visible": True, "reshoot": True}
    if layers:
        on.update(layers)

    fig, ax = plt.subplots(figsize=(10, 10))
    g = result.geometry
    if on["district"]:
        dx, dy = result.district.exterior.xy
        ax.plot(dx, dy, "k-", linewidth=1.5, label="Район")
    if on["visible"]:
        visible_parts = _polygonal_parts(g.visible_union)
        for i, p in enumerate(visible_parts):
            x, y = p.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, facecolor="#2ca02c",
                                    edgecolor="none", alpha=0.55,
                                    label="видимая часть T \\ B" if i == 0 else None))
    if on["objects"]:
        for i, o in enumerate(result.targets.objects):
            x, y = o.geometry.exterior.xy
            ax.plot(x, y, color="black", linewidth=0.7, alpha=0.8,
                    label="объекты T (контуры)" if i == 0 else None)
    if on["obstacles"] and result.obstacles:
        for i, o in enumerate(result.obstacles.objects):
            x, y = o.geometry.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, facecolor="0.7",
                                    edgecolor="0.4", linewidth=0.8, alpha=0.35,
                                    label="препятствия B (облачность)" if i == 0 else None))
    if on["reshoot"]:
        parts = _polygonal_parts(g.reshoot_union)
        for i, p in enumerate(parts):
            x, y = p.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, facecolor="#d62728",
                                    edgecolor="#8b0000", linewidth=0.8, hatch="///",
                                    alpha=0.8,
                                    label="досъёмка РСА: U = T ∩ B" if i == 0 else None))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x, м (локальная СК)")
    ax.set_ylabel("y, м (локальная СК)")
    ax.set_title(
        f"Совместная модель: T — {result.targets.params.model} (seed {result.target_seed}), "
        f"B — {result.obstacles.params.model if result.obstacles else 'нет'} "
        f"(seed {result.obstacle_seed})\n"
        f"P_опт = {g.p_opt:.4f}, P_скр = {g.p_skr:.4f}, "
        f"S(U) = {g.area_reshoot_m2 / 1e6:.2f} км²"
    )
    ax.grid(True, alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
