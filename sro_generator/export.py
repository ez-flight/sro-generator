"""Экспорт сценария: GeoJSON (RFC 7946), CSV, PNG-превью, отчёты JSON."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping

from .generator import Scenario


def _feature(geometry, properties: dict) -> dict:
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def _feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def export_scenario(scenario: Scenario, output_dir: str | Path) -> dict[str, Path]:
    """Записать все файлы результата и вернуть словарь имя -> путь."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = scenario.local
    cfg = scenario.config
    paths: dict[str, Path] = {}

    def dump(name: str, data: dict) -> None:
        p = out / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = p

    # district.geojson — граница района в WGS 84 (lon, lat)
    dump(
        "district.geojson",
        _feature_collection([_feature(local.polygon_to_lonlat(scenario.district_local), {
            "name": cfg.name,
            "center_lat": cfg.district.center_lat,
            "center_lon": cfg.district.center_lon,
            "width_km": cfg.district.width_km,
            "height_km": cfg.district.height_km,
            "rotation_deg": cfg.district.rotation_deg,
        })]),
    )

    # objects.geojson + objects.csv
    obj_features = []
    rows = []
    for o in scenario.objects:
        clon, clat = local.to_lonlat(o.center.x, o.center.y)
        props = {
            "object_id": o.object_id,
            "class_label": o.class_label,
            "area_m2": round(o.area_m2, 3),
            "vertex_count": o.vertex_count,
            "rotation_deg": round(o.rotation_deg, 6),
            "center_lat": clat,
            "center_lon": clon,
            "center_x_m": round(o.center.x, 6),
            "center_y_m": round(o.center.y, 6),
            "priority": o.priority,
            "deadline_h": o.deadline_h,
            "resolution_m": o.resolution_m,
            "placement_mode": o.placement_mode,
            "coordinate_fixed": o.coordinate_fixed,
        }
        obj_features.append(_feature(local.polygon_to_lonlat(o.geometry), props))
        rows.append(props)
    dump("objects.geojson", _feature_collection(obj_features))

    import csv

    csv_path = out / "objects.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paths["objects.csv"] = csv_path

    # frames.geojson
    if scenario.frames:
        feats = []
        for fr in scenario.frames:
            clon, clat = local.to_lonlat(fr.center_x_m, fr.center_y_m)
            feats.append(_feature(local.polygon_to_lonlat(fr.geometry), {
                "frame_id": fr.frame_id,
                "object_id": fr.object_id,
                "pass_azimuth_deg": fr.pass_azimuth_deg,
                "center_lat": clat,
                "center_lon": clon,
                "coverage_fraction": round(fr.coverage_fraction, 6),
            }))
        dump("frames.geojson", _feature_collection(feats))

    # config.resolved.json
    dump("config.resolved.json", cfg.to_dict())

    # generation_report.json
    report = {
        "name": cfg.name,
        "seed": cfg.seed,
        "objects_created": len(scenario.objects),
        "frames_created": len(scenario.frames),
        "placement_attempts_total": scenario.report.attempts_total,
        "warnings": scenario.warnings,
        "checks": {
            "count_matches_config": len(scenario.objects) == cfg.objects.count,
            "all_polygons_valid": all(o.geometry.is_valid and o.geometry.is_simple and o.geometry.area > 0 for o in scenario.objects),
            "areas_within_range": all(
                cfg.objects.area_min_m2 * 0.99 <= o.area_m2 <= cfg.objects.area_max_m2 * 1.01
                for o in scenario.objects
            ),
            "vertex_counts_within_range": all(
                cfg.objects.vertices_min <= o.vertex_count <= cfg.objects.vertices_max for o in scenario.objects
            ),
            "frames_coverage_within_0_1": all(0.0 <= f.coverage_fraction <= 1.0 for f in scenario.frames),
        },
    }
    dump("generation_report.json", report)

    # preview.png
    paths["preview.png"] = render_preview(scenario, out / "preview.png")
    return paths


def render_preview(scenario: Scenario, path: str | Path, show_frames: bool = True, labels: bool = True) -> Path:
    """Нарисовать район, объекты, центры и (опционально) кадры в локальных метрах."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(10, 10))
    dx, dy = scenario.district_local.exterior.xy
    ax.plot(dx, dy, "k-", linewidth=1.5, label="Район")
    colors = plt.cm.tab20.colors
    for i, o in enumerate(scenario.objects):
        x, y = o.geometry.exterior.xy
        patch = MplPolygon(
            list(zip(x, y)), closed=True, facecolor=colors[i % len(colors)],
            edgecolor="black", alpha=0.6, linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.plot(o.center.x, o.center.y, "k+", markersize=6)
        if labels:
            ax.annotate(o.object_id, (o.center.x, o.center.y), fontsize=7, alpha=0.8)
    if show_frames:
        for fr in scenario.frames:
            x, y = fr.geometry.exterior.xy
            ax.plot(x, y, "r--", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("X, м (восток)")
    ax.set_ylabel("Y, м (север)")
    ax.set_title(f"Район «{scenario.config.name}»: {len(scenario.objects)} объектов, seed={scenario.config.seed}")
    margin_x = (max(dx) - min(dx)) * 0.04
    margin_y = (max(dy) - min(dy)) * 0.04
    ax.set_xlim(min(dx) - margin_x, max(dx) + margin_x)
    ax.set_ylim(min(dy) - margin_y, max(dy) + margin_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
