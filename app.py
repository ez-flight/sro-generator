"""Графический интерфейс генератора района СРО (Streamlit).

Запуск: streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.export import render_preview
from sro_generator.generator import generate_scenario
from sro_generator.placement import PlacementError

st.set_page_config(page_title="Генератор района СРО", layout="wide")
st.title("Генератор района со случайно расположенными объектами (СРО)")

with st.sidebar:
    st.header("Район")
    name = st.text_input("Имя сценария", "random_district_demo")
    center_lat = st.number_input("Центр: широта", value=55.0, format="%.6f")
    center_lon = st.number_input("Центр: долгота", value=60.0, format="%.6f")
    width_unit = st.selectbox("Единицы размера", ["км", "м"])
    factor = 1.0 if width_unit == "км" else 0.001
    width = st.number_input(f"Ширина района, {width_unit}", value=100.0, min_value=0.001) * factor
    height = st.number_input(f"Высота района, {width_unit}", value=90.0, min_value=0.001) * factor
    rotation = st.number_input("Азимут поворота района, град", value=0.0)

    st.header("Объекты")
    count = st.number_input("Число объектов", value=25, min_value=1, step=1)
    col1, col2 = st.columns(2)
    vertices_min = col1.number_input("Вершин мин", value=3, min_value=3, max_value=10, step=1)
    vertices_max = col2.number_input("Вершин макс", value=10, min_value=3, max_value=10, step=1)
    area_unit = st.selectbox("Единицы площади", ["км²", "га", "м²"])
    area_factor = {"км²": 1e6, "га": 1e4, "м²": 1.0}[area_unit]
    area_min = st.number_input(f"Площадь мин, {area_unit}", value=0.5, min_value=0.000001) * area_factor
    area_max = st.number_input(f"Площадь макс, {area_unit}", value=5.0, min_value=0.000001) * area_factor
    irregularity = st.slider("Нерегулярность формы (0–1)", 0.0, 1.0, 0.35)
    rot_min = st.number_input("Поворот объекта мин, град", value=0.0)
    rot_max = st.number_input("Поворот объекта макс, град", value=360.0)
    max_aspect = st.number_input("Предельное отношение сторон", value=2.5, min_value=1.01)

    st.header("Размещение")
    dist_unit = st.selectbox("Единицы расстояний", ["м", "км"])
    dist_factor = 1.0 if dist_unit == "м" else 1000.0
    min_distance = st.number_input(f"Мин. расстояние между объектами, {dist_unit}", value=500.0) * dist_factor
    boundary_margin = st.number_input(f"Отступ от границы, {dist_unit}", value=100.0) * dist_factor
    allow_overlap = st.checkbox("Разрешить пересечения", value=False)
    max_overlap = st.slider("Макс. доля перекрытия", 0.0, 1.0, 0.0, disabled=not allow_overlap)
    max_attempts = st.number_input("Макс. попыток на объект", value=10000, min_value=1, step=1000)

    st.header("Атрибуты")
    class_labels = st.multiselect("Категории объектов", ["warehouse", "industrial", "transport", "energy", "agro"],
                                  default=["warehouse", "industrial", "transport"])
    pr_min = st.number_input("Приоритет мин", value=1, min_value=1, step=1)
    pr_max = st.number_input("Приоритет макс", value=5, min_value=1, step=1)
    deadline_choices = st.multiselect("Директивные сроки, ч", [12, 24, 30, 48, 72], default=[30, 48])
    resolution_choices = st.multiselect("Разрешение, м", [0.3, 0.5, 1.0, 2.0, 5.0], default=[0.5, 1.0])

    st.header("Кадры РСА")
    frames_enabled = st.checkbox("Строить кадры-кандидаты", value=True)
    along_km = st.number_input("Размер вдоль трассы, км", value=5.0, min_value=0.1)
    cross_km = st.number_input("Размер поперёк трассы, км", value=5.0, min_value=0.1)
    azimuths = st.text_input("Азимуты пролёта, град (через запятую)", "194.0, 346.0")

    st.header("Воспроизводимость")
    seed = st.number_input("Seed", value=42, step=1)

st.subheader("Ручные центры (пусто — полностью случайное размещение)")
centers_df = st.data_editor(
    pd.DataFrame(columns=["object_id", "latitude", "longitude"]),
    num_rows="dynamic",
    key="manual_centers",
)

show_frames = st.checkbox("Показывать кадры на рисунке", value=True) if frames_enabled else False
show_labels = st.checkbox("Подписи object_id", value=True)

if st.button("Сгенерировать", type="primary"):
    manual_centers = []
    errors = []
    seen = set()
    for idx, row in centers_df.iterrows():
        oid = str(row.get("object_id") or "").strip()
        lat, lon = row.get("latitude"), row.get("longitude")
        if not oid and pd.isna(lat) and pd.isna(lon):
            continue
        if not oid:
            errors.append(f"Строка {idx + 1}: заполните object_id.")
            continue
        if oid in seen:
            errors.append(f"Строка {idx + 1}: дублирующийся object_id «{oid}».")
            continue
        seen.add(oid)
        try:
            manual_centers.append({"object_id": oid, "latitude": float(lat), "longitude": float(lon)})
        except (TypeError, ValueError):
            errors.append(f"Строка {idx + 1}: latitude/longitude должны быть числами.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    config_dict = {
        "name": name,
        "seed": int(seed),
        "district": {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "width_km": width,
            "height_km": height,
            "rotation_deg": rotation,
        },
        "objects": {
            "count": int(count),
            "vertices_min": int(vertices_min),
            "vertices_max": int(vertices_max),
            "area_min_m2": area_min,
            "area_max_m2": area_max,
            "irregularity": irregularity,
            "rotation_min_deg": rot_min,
            "rotation_max_deg": rot_max,
            "max_aspect_ratio": max_aspect,
        },
        "placement": {
            "mode": "random",
            "min_distance_m": min_distance,
            "boundary_margin_m": boundary_margin,
            "allow_overlap": allow_overlap,
            "max_overlap_fraction": max_overlap if allow_overlap else 0.0,
            "max_attempts_per_object": int(max_attempts),
            "manual_centers": manual_centers,
        },
        "attributes": {
            "class_labels": class_labels,
            "priority_min": int(pr_min),
            "priority_max": int(pr_max),
            "deadline_choices_h": [float(d) for d in deadline_choices] or [30.0],
            "resolution_choices_m": [float(r) for r in resolution_choices] or [1.0],
        },
        "frames": {
            "enabled": frames_enabled,
            "along_track_km": along_km,
            "cross_track_km": cross_km,
            "pass_azimuths_deg": [float(a) for a in azimuths.split(",") if a.strip()] if frames_enabled else [],
        },
    }
    try:
        config = ScenarioConfig.model_validate(config_dict)
    except ConfigError as exc:
        st.error(f"Ошибка конфигурации: {exc}")
        st.stop()

    with st.spinner("Генерация…"):
        try:
            scenario = generate_scenario(config)
        except PlacementError as exc:
            st.error(f"Ошибка размещения: {exc}")
            st.stop()

    preview_path = render_preview(scenario, "/tmp/sro_preview.png", show_frames=show_frames, labels=show_labels)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.image(str(preview_path), caption=f"Район «{config.name}», seed={config.seed}")

    areas = [o.area_m2 for o in scenario.objects]
    manual_share = sum(1 for o in scenario.objects if o.coordinate_fixed) / len(scenario.objects)
    with col_r:
        st.subheader("Сводка")
        st.metric("Число объектов", len(scenario.objects))
        st.metric("Кадров", len(scenario.frames))
        st.metric("Площадь объектов, м²", f"{min(areas):,.0f} – {max(areas):,.0f}")
        st.metric("Вершин (мин–макс)", f"{min(o.vertex_count for o in scenario.objects)} – {max(o.vertex_count for o in scenario.objects)}")
        st.metric("Доля ручных центров", f"{manual_share:.0%}")
        st.metric("Попыток размещения", scenario.report.attempts_total)
        for w in scenario.warnings:
            st.warning(w)

    st.subheader("Атрибуты объектов")
    st.dataframe(pd.DataFrame([
        {
            "object_id": o.object_id,
            "class_label": o.class_label,
            "area_m2": round(o.area_m2, 1),
            "vertex_count": o.vertex_count,
            "priority": o.priority,
            "deadline_h": o.deadline_h,
            "resolution_m": o.resolution_m,
            "placement_mode": o.placement_mode,
            "coordinate_fixed": o.coordinate_fixed,
            "center_lat": round(scenario.local.to_lonlat(o.center.x, o.center.y)[1], 6),
            "center_lon": round(scenario.local.to_lonlat(o.center.x, o.center.y)[0], 6),
        }
        for o in scenario.objects
    ]))

    buf = io.BytesIO()
    with open(preview_path, "rb") as f:
        buf.write(f.read())
    st.download_button("Скачать preview.png", buf.getvalue(), file_name="preview.png", mime="image/png")
    st.session_state["scenario"] = scenario
