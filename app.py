"""Графический интерфейс SRO Generator 2.4.0 (Streamlit).

Пять вкладок:
1. «Генерация СРО» — настройка района и объектов, воспроизводимая генерация
   (seed), визуализация района, объектов, облачности и кадров досъёмки РСА,
   расчёт геометрической полноты покрытия (G — район, C — скрытые объекты).
2. «Статистическое исследование» — оценивание математического ожидания и
   дисперсии геометрической полноты покрытия района методом статистических
   испытаний (M*(n), D̃(n), σ̃(n), ε(β,n), пять обязательных графиков).
3. «Модель объектов съёмки» — четыре закона пространственного размещения
   целевых объектов (равномерно-случайное, групповое, регулярное, линейное),
   визуализация, геометрические характеристики, экспорт GeoJSON/JSON
   (sro_generator/target_model.py).
4. «Статистическое исследование объектов съёмки» — проверка законов
   размещения: n независимых реализаций каждого закона, F*(r) с имитационным
   эталоном и кривой Рэлея, сравнение по R/I/R_b, сходимость M*(n), D̃(n),
   ε(β,n), две таблицы, шесть обязательных графиков (PNG/SVG).
5. «Совместная модель района ДЗЗ» — объекты T и препятствия оптической
   съёмке B на одном районе с независимыми seed; U = T ∩ B, P_опт, P_скр,
   классификация видимости, карта слоёв и экспорт GeoJSON/JSON
   (sro_generator/joint_model.py).

Исследовательские стенды — в benchmarks/ (convergence_analysis.py,
monte_carlo_convergence.py, cloud_coverage_analysis.py); вычислительные
функции сохранены в sro_generator/convergence.py, sro_generator/monte_carlo.py
и sro_generator/target_model.py.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.generator import generate_scenario
from sro_generator.geometry import LocalCRS, build_district_polygon
from sro_generator.monte_carlo import (
    classical_series,
    k_sweep,
    render_classical_charts,
    required_trials,
    run_monte_carlo_experiment,
    MonteCarloParams,
)
from sro_generator.placement import PlacementError
from sro_generator.joint_model import (
    export_joint_geojson,
    generate_joint_model,
    joint_summary,
    render_joint_map,
)
from sro_generator.target_model import (
    MODEL_TITLES_RU,
    MODELS,
    TARGET_MODELS,
    SpatialModelParams,
    collect_nn_samples,
    export_target_geojson,
    generate_target_objects,
    load_curve_geojson,
    realization_metrics,
    render_statistics_charts,
    run_target_model_study,
    sample_skewness,
)

st.set_page_config(page_title="SRO Generator 2.4", layout="wide")
st.title("SRO Generator 2.4 — генерация СРО, пространственные модели и досъёмка РСА")

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

    st.header("Облачность и досъёмка")
    cloud_fraction = st.slider("Доля объектов, недоступных оптике (облачность)", 0.0, 1.0, 0.4)

    st.header("Воспроизводимость")
    seed = st.number_input("Seed (базовый)", value=42, step=1)

    st.header("Вычисления")
    workers = st.number_input(
        "Параллельных потоков",
        value=min(4, os.cpu_count() or 1),
        min_value=1, max_value=32, step=1,
    )


def parse_manual_centers(df) -> tuple[list[dict], list[str]]:
    """Разобрать таблицу ручных центров; вернуть (центры, ошибки)."""
    manual_centers, errors, seen = [], [], set()
    for idx, row in df.iterrows():
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
    return manual_centers, errors


def build_config(manual_centers: list[dict]) -> dict:
    """Собрать конфигурацию из параметров сайдбара."""
    return {
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


def render_preview_cloud(scenario, hidden_flags, frames_selected, path, show_frames=True, labels=True):
    """Район, объекты (скрытые от оптики — серым со штриховкой), кадры досъёмки."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(10, 10))
    dx, dy = scenario.district_local.exterior.xy
    ax.plot(dx, dy, "k-", linewidth=1.5, label="Район")
    colors = plt.cm.tab20.colors
    sel_ids = set(frames_selected)
    for i, o in enumerate(scenario.objects):
        x, y = o.geometry.exterior.xy
        if hidden_flags[i]:
            patch = MplPolygon(list(zip(x, y)), closed=True, facecolor="0.75",
                               edgecolor="0.3", hatch="///", alpha=0.85,
                               label="объект недоступен оптике" if i == int(np.argmax(hidden_flags)) else None)
        else:
            patch = MplPolygon(list(zip(x, y)), closed=True, facecolor=colors[i % len(colors)],
                               edgecolor="black", linewidth=0.6, alpha=0.8,
                               label="объект доступен оптике" if not any(hidden_flags[:i]) else None)
        ax.add_patch(patch)
        if labels:
            cx, cy = o.center.x, o.center.y
            ax.annotate(o.object_id, (cx, cy), fontsize=7, ha="center", va="center")
    if show_frames:
        for f in scenario.frames:
            x, y = f.geometry.exterior.xy
            selected = f.frame_id in sel_ids
            ax.plot(x, y, color="#d62728" if selected else "#1f77b4",
                    linewidth=1.8 if selected else 0.7,
                    linestyle="-" if selected else "--",
                    alpha=0.95 if selected else 0.35,
                    label="кадр досъёмки (план)" if selected and f.frame_id == frames_selected[0] else
                          ("кадр-кандидат" if f.frame_id == scenario.frames[0].frame_id else None))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x, м (локальная СК)")
    ax.set_ylabel("y, м (локальная СК)")
    ax.set_title(f"Район «{scenario.config.name}», seed={scenario.config.seed}")
    ax.grid(True, alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


tab_gen, tab_stat, tab_model, tab_model_stat, tab_joint = st.tabs(
    ["Генерация СРО", "Статистическое исследование", "Модель объектов съёмки",
     "Статистическое исследование объектов съёмки", "Совместная модель района ДЗЗ"])

# ---------------------------------------------------------------------------
# Вкладка 1. Генерация СРО
# ---------------------------------------------------------------------------
with tab_gen:
    st.subheader("Генерация района со случайными объектами (СРО)")
    st.caption(
        "Один прогон: детерминированная генерация из seed. Облачность — независимая "
        "бинарная маска «объект недоступен оптике»; кадры РСА — план досъёмки "
        "(запланированные наблюдения, не выполненные съёмки). Красным выделены "
        "кадры плана, серым со штриховкой — объекты под облачностью."
    )
    centers_df = st.data_editor(
        pd.DataFrame(columns=["object_id", "latitude", "longitude"]),
        num_rows="dynamic",
        key="manual_centers",
    )
    gen_k = st.number_input("Итераций досъёмки K", value=30, min_value=1, max_value=500, step=5,
                            key="gen_k")
    gen_sel = st.selectbox("Алгоритм досъёмки", ["greedy", "softmax"], key="gen_sel")
    show_frames = st.checkbox("Показывать кадры на рисунке", value=True) if frames_enabled else False
    show_labels = st.checkbox("Подписи object_id", value=True)

    if st.button("Сгенерировать", type="primary"):
        manual_centers, errors = parse_manual_centers(centers_df)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
        config_dict = build_config(manual_centers)
        try:
            config = ScenarioConfig.model_validate(config_dict)
        except ConfigError as exc:
            st.error(f"Ошибка конфигурации: {exc}")
            st.stop()

        with st.spinner("Генерация и расчёт полноты…"):
            try:
                scenario = generate_scenario(config)
            except PlacementError as exc:
                st.error(f"Ошибка размещения: {exc}")
                st.stop()
            if frames_enabled:
                params = MonteCarloParams(
                    experiments=2, iterations=int(gen_k), checkpoints=(2,),
                    cloud_fraction=float(cloud_fraction), selection=gen_sel,
                )
                exp = run_monte_carlo_experiment(config, params, int(seed))
                scenario2 = generate_scenario(
                    ScenarioConfig.model_validate({**config_dict, "seed": exp["seed"]})
                )
                mask_rng = np.random.default_rng(np.random.SeedSequence([exp["seed"], 20261126]))
                hidden = list(mask_rng.random(len(scenario2.objects)) < cloud_fraction)
                preview = render_preview_cloud(scenario2, hidden, exp["frames_selected"],
                                               "/tmp/sro_preview.png", show_frames, show_labels)
                g_val, c_val = exp["final_geometric_completeness"], exp["hidden_objects_coverage"]
                optical = exp["optical_district_coverage"]
            else:
                hidden = [False] * len(scenario.objects)
                preview = render_preview_cloud(scenario, hidden, [],
                                               "/tmp/sro_preview.png", False, show_labels)
                g_val = c_val = optical = None

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.image(str(preview), caption=f"Район «{config.name}», seed={config.seed}")
        areas = [o.area_m2 for o in scenario.objects]
        with col_r:
            st.subheader("Сводка")
            st.metric("Число объектов", len(scenario.objects))
            st.metric("Кадров-кандидатов", len(scenario.frames))
            st.metric("Площадь объектов, м²", f"{min(areas):,.0f} – {max(areas):,.0f}")
            st.metric("Вершин (мин–макс)", f"{min(o.vertex_count for o in scenario.objects)} – "
                                            f"{max(o.vertex_count for o in scenario.objects)}")
            st.metric("Под облачностью", f"{sum(hidden)} из {len(hidden)}")
            if g_val is not None:
                st.metric("Оптическое покрытие района", f"{optical:.4f}")
                st.metric("Полнота досъёмки скрытых C", f"{c_val:.4f}")
                st.metric("Итоговая полнота района G", f"{g_val:.4f}")
            st.metric("Доля ручных центров",
                      f"{sum(1 for o in scenario.objects if o.coordinate_fixed) / len(scenario.objects):.0%}")
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
        with open(preview, "rb") as f:
            buf.write(f.read())
        st.download_button("Скачать preview.png", buf.getvalue(), file_name="preview.png",
                           mime="image/png")

# ---------------------------------------------------------------------------
# Вкладка 2. Статистическое исследование
# ---------------------------------------------------------------------------
with tab_stat:
    st.subheader("Оценивание математического ожидания и дисперсии геометрической "
                 "полноты покрытия района при досъёмке средствами РСА ДЗЗ "
                 "методом статистических испытаний")
    st.caption(
        "Каждое испытание — новая случайная конфигурация объектов СРО со своим seed; "
        "облачность скрывает часть объектов от оптики, алгоритм досъёмки РСА строит "
        "план из K кадров. Реализация случайной величины xᵢ — геометрическая полнота "
        "покрытия G итогового плана. Графики 1–4 — сходимость статистических оценок "
        "при росте объёма испытаний n (фиксированный K); график 5 — сходимость "
        "алгоритма планирования при росте числа итераций K (фиксированный n). "
        "K — число итераций алгоритма, не число пролётов КА."
    )

    sc1, sc2, sc3 = st.columns(3)
    n_trials = sc1.number_input("Число статистических испытаний n", value=200, min_value=3,
                                max_value=5000, step=50)
    beta = sc2.selectbox("Надёжность β", [0.90, 0.95, 0.99], index=1)
    epsilon0 = sc3.number_input("Заданная точность ε₀", value=0.01, min_value=0.0001,
                                format="%.4f")
    sc4, sc5, sc6 = st.columns(3)
    k_fixed = sc4.number_input("Число итераций алгоритма K", value=70, min_value=1, max_value=300, step=5)
    selection = sc5.selectbox("Алгоритм досъёмки", ["greedy", "softmax"], key="stat_sel")
    k_grid = sc6.multiselect("Сетка K для графика 5", [5, 10, 20, 30, 50, 70],
                             default=[5, 10, 20, 30, 50, 70])
    if not k_grid:
        st.error("Выберите хотя бы одно значение K для сетки.")
        st.stop()
    k_values = tuple(sorted({int(k) for k in k_grid} | {int(k_fixed)}))

    if st.button("Провести статистические испытания", type="primary"):
        config_dict = build_config([])
        try:
            config = ScenarioConfig.model_validate(config_dict)
        except ConfigError as exc:
            st.error(f"Ошибка конфигурации: {exc}")
            st.stop()
        if not config.frames.enabled:
            st.error("Включите кадры РСА (панель «Кадры РСА» слева).")
            st.stop()

        progress = st.progress(0.0, text="Статистические испытания…")
        sweep = k_sweep(
            config,
            cloud_fraction=float(cloud_fraction),
            k_values=k_values,
            selection=selection,
            experiments=int(n_trials),
            seed_start=0,
            beta=float(beta),
            workers=int(workers),
            keep_runs=True,
            progress=lambda done, total: progress.progress(
                done / total, text=f"Испытаний завершено: {done}/{total}"),
        )
        progress.empty()
        st.session_state["stat_sweep"] = sweep
        st.session_state["stat_k_fixed"] = int(k_fixed)
        st.session_state["stat_beta"] = float(beta)
        st.session_state["stat_eps0"] = float(epsilon0)

    sweep = st.session_state.get("stat_sweep")
    if sweep:
        k_fixed = st.session_state["stat_k_fixed"]
        beta = st.session_state["stat_beta"]
        epsilon0 = st.session_state["stat_eps0"]
        if k_fixed not in sweep:
            st.error(f"K = {k_fixed} отсутствует в сетке исследования.")
            st.stop()
        g_values = [r["final_geometric_completeness"] for r in sweep[k_fixed]["runs"]]
        series = classical_series(g_values, beta=beta)
        n_req = required_trials(series[-1]["sigma_tilde"], epsilon0, beta)

        out_dir = tempfile.mkdtemp(prefix="sro_stat_")
        chart_paths = render_classical_charts(series, sweep, beta, epsilon0, out_dir)

        st.markdown("**Итоговые оценки** "
                    f"(n = {series[-1]['n']}, β = {beta:.2f}, K = {k_fixed}, "
                    f"облачность {cloud_fraction:.0%}, алгоритм {selection})")
        final = series[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("M*(n)", f"{final['m_star']:.4f}")
        m2.metric("D̃(n)", f"{final['d_tilde']:.3g}")
        m3.metric("σ̃(n)", f"{final['sigma_tilde']:.4f}")
        m4.metric("ε(β,n)", f"{final['epsilon']:.4f}")
        m5.metric("n для ε₀", f"≥ {n_req['n_required']}")
        st.caption(
            f"Доверительный интервал математического ожидания: "
            f"I(β={beta:.2f}, n) = [{final['ci_low']:.4f}; {final['ci_high']:.4f}]. "
            f"Потребный объём испытаний (пилотная нормальная оценка): "
            f"n ≥ (z·σ̃/ε₀)² = {n_req['n_required']}. {n_req['note']}"
        )

        for p in chart_paths.values():
            st.image(str(p))

        st.subheader("Таблица оценок")
        step = max(1, len(series) // 20)
        st.dataframe(pd.DataFrame([
            {
                "n": r["n"],
                "M*(n)": round(r["m_star"], 6),
                "D̃(n)": round(r["d_tilde"], 8),
                "σ̃(n)": round(r["sigma_tilde"], 6),
                "ε(β,n)": round(r["epsilon"], 6),
                "I нижн.": round(r["ci_low"], 6),
                "I верхн.": round(r["ci_high"], 6),
            }
            for r in series[::step] + [series[-1]]
        ]))
        st.caption(
            "M*(n) — статистическое математическое ожидание; D̃(n) — исправленная "
            "выборочная дисперсия (деление на n−1); σ̃(n) = √D̃; "
            "ε(β,n) = t(β,n−1)·σ̃(n)/√n; I = [M*−ε; M*+ε]. Данные не сглаживаются."
        )

# ---------------------------------------------------------------------------
# Вкладка 3. Модель объектов съёмки
# ---------------------------------------------------------------------------

def render_target_preview(result, path):
    """Район, объекты модели и структура закона (центры групп / линейные хорды)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(10, 10))
    dx, dy = result.district.exterior.xy
    ax.plot(dx, dy, "k-", linewidth=1.5, label="Район")
    structure_label = ("криволинейная структура" if result.params.model == "curved"
                       else "линейная структура")
    for chord in result.line_chords:
        cx, cy = chord.xy
        ax.plot(cx, cy, color="0.4", linestyle="--", linewidth=1.2,
                label=structure_label if chord is result.line_chords[0] else None)
    for i, c in enumerate(result.cluster_centers):
        ax.plot(c.x, c.y, marker="x", color="red", markersize=10, markeredgewidth=2.2,
                label="центр группы" if i == 0 else None)
    colors = plt.cm.tab20.colors
    for i, o in enumerate(result.objects):
        x, y = o.geometry.exterior.xy
        face = colors[o.group_id % len(colors)] if o.group_id is not None else colors[i % len(colors)]
        ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, facecolor=face,
                                edgecolor="black", linewidth=0.6, alpha=0.75,
                                label="объект съёмки" if i == 0 else None))
        ax.plot(o.center.x, o.center.y, "k+", markersize=5)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x, м (локальная СК)")
    ax.set_ylabel("y, м (локальная СК)")
    ax.set_title(f"{MODEL_TITLES_RU[result.params.model]} размещение, "
                 f"{len(result.objects)} объектов")
    ax.grid(True, alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


with tab_model:
    st.subheader("Модель объектов съёмки: законы пространственного размещения")
    st.caption(
        "Математическая модель пространственного распределения целевых объектов ДЗЗ "
        "в районе (панель «Район» и «Объекты» слева). Четыре базовых закона: "
        "равномерно-случайное, групповое (кластерное), регулярное (решётка с дрожанием) "
        "и линейное (вдоль прямых структур), а также криволинейное расширение линейного "
        "закона — вдоль случайной или импортированной из GeoJSON плавной кривой. Объекты "
        "имеют геометрию, размеры, площадь и координаты; соблюдаются границы района с "
        "отступом и минимальное расстояние; реализация воспроизводима по seed."
    )

    law_labels = {m: MODEL_TITLES_RU[m] for m in TARGET_MODELS}
    tm_model = st.selectbox(
        "Закон пространственного распределения", list(TARGET_MODELS),
        format_func=lambda m: law_labels[m], key="tm_model")
    tc1, tc2, tc3 = st.columns(3)
    tm_n_clusters = tc1.number_input("Число групп (кластерное)", value=5, min_value=1, step=1,
                                     disabled=tm_model != "cluster")
    tm_cluster_sigma = tc2.number_input("СКО группы, м (кластерное)", value=1500.0, min_value=1.0,
                                        disabled=tm_model != "cluster")
    tm_jitter = tc3.slider("Дрожание решётки (регулярное)", 0.0, 1.0, 0.6,
                           disabled=tm_model != "regular")
    tc4, tc5, tc6 = st.columns(3)
    tm_n_lines = tc4.number_input("Число прямых структур", value=4, min_value=1, step=1,
                                  disabled=tm_model != "linear")
    tm_line_sigma = tc5.number_input("СКО поперечного отклонения, м (линейное)", value=400.0,
                                     min_value=0.0, disabled=tm_model != "linear")
    tm_curve_bend = tc6.number_input("Извилистость кривой, м", value=2500.0, min_value=0.0,
                                     disabled=tm_model != "curved")
    tc7, tc8, _ = st.columns(3)
    tm_curve_width = tc7.number_input("Ширина коридора, м (криволинейное)", value=1000.0,
                                      min_value=0.0, disabled=tm_model != "curved")
    tm_curve_file = st.file_uploader(
        "Импорт кривой из GeoJSON (LineString/MultiLineString, WGS 84)",
        type=["geojson", "json"], disabled=tm_model != "curved", key="tm_curve_file")
    if tm_model == "curved":
        st.caption("Если файл не загружен, кривая строится случайно от хорды района. "
                   "Объекты выбираются равномерно по длине кривой в пределах коридора.")

    tm_curve_bytes = tm_curve_file.getvalue() if tm_curve_file is not None else None
    tm_curve_hash = (hashlib.sha256(tm_curve_bytes).hexdigest()
                     if tm_curve_bytes is not None else None)
    tm_controls = (
        float(center_lat), float(center_lon), float(width), float(height), float(rotation),
        int(count), int(vertices_min), int(vertices_max), float(area_min), float(area_max),
        float(irregularity), float(rot_min), float(rot_max), float(max_aspect),
        float(min_distance), float(boundary_margin), int(max_attempts), int(seed),
        str(tm_model), int(tm_n_clusters), float(tm_cluster_sigma), float(tm_jitter),
        int(tm_n_lines), float(tm_line_sigma), float(tm_curve_bend), float(tm_curve_width),
        tm_curve_hash,
    )
    if st.session_state.get("tm_controls") != tm_controls:
        if st.session_state.get("tm_result") is not None:
            st.info("Параметры модели объектов изменились. Нажмите «Разместить объекты».")
        st.session_state["tm_result"] = None
        st.session_state["tm_metrics"] = None
        st.session_state["tm_preview"] = None
        st.session_state["tm_local"] = None
        st.session_state["tm_controls"] = tm_controls

    if st.button("Разместить объекты", type="primary", key="tm_run"):
        # Не оставляем прежний результат при ошибке параметров/размещения/импорта.
        st.session_state["tm_result"] = None
        st.session_state["tm_metrics"] = None
        st.session_state["tm_preview"] = None
        st.session_state["tm_local"] = None
        local = LocalCRS(center_lat, center_lon)
        district = build_district_polygon(local, width * 1000.0, height * 1000.0, rotation)
        tm_curve_line = None
        if tm_curve_bytes is not None:
            try:
                tm_curve_line = load_curve_geojson(json.loads(tm_curve_bytes.decode("utf-8-sig")), local)
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Ошибка импорта кривой: {exc}")
                st.stop()
        try:
            tm_params = SpatialModelParams(
                model=tm_model,
                count=int(count),
                vertices_min=int(vertices_min),
                vertices_max=int(vertices_max),
                area_min_m2=float(area_min),
                area_max_m2=float(area_max),
                irregularity=float(irregularity),
                max_aspect_ratio=float(max_aspect),
                rotation_min_deg=float(rot_min),
                rotation_max_deg=float(rot_max),
                boundary_margin_m=float(boundary_margin),
                min_distance_m=float(min_distance),
                max_attempts_per_object=int(max_attempts),
                n_clusters=int(tm_n_clusters),
                cluster_sigma_m=float(tm_cluster_sigma),
                jitter_fraction=float(tm_jitter),
                n_lines=int(tm_n_lines),
                line_sigma_m=float(tm_line_sigma),
                curve_bend_m=float(tm_curve_bend),
                curve_width_m=float(tm_curve_width),
            )
        except ValueError as exc:
            st.error(f"Ошибка параметров: {exc}")
            st.stop()
        with st.spinner("Размещение объектов…"):
            try:
                tm_result = generate_target_objects(
                    tm_params, district, np.random.default_rng(int(seed)), curve=tm_curve_line)
            except PlacementError as exc:
                st.error(f"Ошибка размещения: {exc}")
                st.stop()
            tm_metrics = realization_metrics(tm_result)
            tm_preview = render_target_preview(tm_result, "/tmp/sro_target_preview.png")
        st.session_state["tm_result"] = tm_result
        st.session_state["tm_metrics"] = tm_metrics
        st.session_state["tm_preview"] = tm_preview
        st.session_state["tm_local"] = local
        st.session_state["tm_curve_source"] = "импортированная линия" if tm_curve_line is not None else "случайная кривая"

    tm_result = st.session_state.get("tm_result")
    if tm_result is not None:
        tm_metrics = st.session_state["tm_metrics"]
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.image(st.session_state["tm_preview"],
                     caption=f"{MODEL_TITLES_RU[tm_result.params.model]} размещение")
        with col_r:
            st.subheader("Сводка")
            st.metric("Число объектов", tm_metrics["n_objects"])
            if tm_result.params.model == "curved":
                st.metric("Источник кривой", st.session_state.get("tm_curve_source", "случайная кривая"))
            st.metric("Плотность, объектов/км²", f"{tm_metrics['density_per_km2']:.3f}")
            st.metric("Доля площади района", f"{tm_metrics['objects_area_fraction']:.4f}")
            st.metric("Средняя площадь объекта, м²", f"{tm_metrics['objects_area_mean_m2']:,.0f}")
            st.metric("Среднее расстояние до ближайшего соседа, м",
                      f"{tm_metrics['nn_mean_m']:,.0f}")
            st.metric("Индекс Кларка–Эванса R", f"{tm_metrics['ce_r']:.3f}")
            st.metric("Индекс дисперсии квадрантов I", f"{tm_metrics['dispersion_index']:.3f}")
            st.caption(
                f"Диагностика реализации: R < 1 — группирование, R ≈ 1 — случайное, "
                f"R > 1 — регулярное; I — то же по квадрантному счёту. "
                f"χ²-критерий CSR: {'отвергается' if tm_metrics['quad_csr_rejected'] else 'не отвергается'} "
                f"(χ² = {tm_metrics['quad_chi2']:.1f}, критические "
                f"{tm_metrics['quad_chi2_low_crit']:.1f}…{tm_metrics['quad_chi2_high_crit']:.1f})."
            )

        st.subheader("Геометрические характеристики объектов")
        st.dataframe(pd.DataFrame([
            {
                "object_id": o.object_id,
                "group_id": o.group_id,
                "area_m2": round(o.area_m2, 1),
                "vertex_count": o.vertex_count,
                "rotation_deg": round(o.rotation_deg, 2),
                "center_x_m": round(o.center.x, 1),
                "center_y_m": round(o.center.y, 1),
            }
            for o in tm_result.objects
        ]))

        import json as _json

        fc = export_target_geojson(tm_result, st.session_state["tm_local"])
        dc1, dc2, _ = st.columns(3)
        dc1.download_button(
            "Скачать objects.geojson",
            _json.dumps(fc, ensure_ascii=False),
            file_name="target_objects.geojson",
            mime="application/geo+json",
        )
        dc2.download_button(
            "Скачать параметры модели (JSON)",
            _json.dumps({
                "model": tm_result.params.model,
                "model_title_ru": MODEL_TITLES_RU[tm_result.params.model],
                "params": {k: v for k, v in vars(tm_result.params).items()},
                "seed": int(seed),
                "district": {
                    "center_lat": center_lat, "center_lon": center_lon,
                    "width_km": width, "height_km": height, "rotation_deg": rotation,
                },
                "metrics": {k: v for k, v in tm_metrics.items()
                            if isinstance(v, (int, float, bool, str))},
            }, ensure_ascii=False, indent=2),
            file_name="target_model_params.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Вкладка 4. Статистическое исследование объектов съёмки
# ---------------------------------------------------------------------------
with tab_model_stat:
    st.subheader("Статистическое исследование объектов съёмки: проверка законов "
                 "пространственного распределения и сходимость оценок")
    st.caption(
        "Одно статистическое испытание — одна независимая реализация пространственного "
        "распределения объектов (модель — вкладка «Модель объектов съёмки», район и объекты — "
        "панель слева). Исследование: n независимых реализаций каждого из четырёх законов; "
        "классические оценки M*(n), D̃(n), σ̃(n), ε(β,n) и доверительные интервалы. "
        "График 2 сопоставляет эмпирическую F*(r) расстояний до ближайшего соседа с "
        "независимым имитационным эталоном (другие seed) и аналитической кривой Рэлея для "
        "случайного размещения (CSR). Расстояния — характеристики отдельных объектов внутри "
        "реализации (зависимые): используются для формы распределения, а не для ДИ среднего."
    )

    ts1, ts2, ts3, ts4 = st.columns(4)
    ts_n = ts1.number_input("Число испытаний n (реализаций на закон)", value=200, min_value=20,
                            max_value=1000, step=20, key="ts_n")
    ts_beta = ts2.selectbox("Надёжность β", [0.90, 0.95, 0.99], index=1, key="ts_beta")
    ts_eps0 = ts3.number_input("Заданная точность ε₀, м", value=5.0, min_value=0.01,
                               format="%.2f", key="ts_eps0")
    ts_seed = ts4.number_input("Начальный seed испытаний", value=0, step=1, key="ts_seed")

    ts5, ts6, ts7, ts8 = st.columns(4)
    ts_model = ts5.selectbox("Закон для графиков 4–6", list(MODELS),
                             format_func=lambda m: MODEL_TITLES_RU[m], key="ts_model")
    ts_metric = ts6.selectbox(
        "Пространственная характеристика",
        ["nn_mean_m", "ce_r", "dispersion_index", "nn_bearing_r", "objects_area_fraction"],
        format_func=lambda k: {
            "nn_mean_m": "Среднее расстояние до ближайшего соседа, м",
            "ce_r": "Индекс Кларка–Эванса R",
            "dispersion_index": "Индекс квадрантной дисперсии I",
            "nn_bearing_r": "Направленность R_b",
            "objects_area_fraction": "Доля площади района",
        }[k], key="ts_metric")
    ts_clusters = ts7.number_input("Групп (кластерный закон)", value=5, min_value=1, step=1,
                                   key="ts_clusters")
    ts_sigma = ts8.number_input("СКО группы, м", value=1500.0, min_value=1.0, key="ts_sigma")
    ts9, ts10, _, _ = st.columns(4)
    ts_jitter = ts9.slider("Дрожание решётки (регулярный)", 0.0, 1.0, 0.6, key="ts_jitter")
    ts_lines = ts10.number_input("Линейных структур", value=4, min_value=1, step=1, key="ts_lines")

    if st.button("Провести исследование", type="primary", key="ts_run"):
        local = LocalCRS(center_lat, center_lon)
        district = build_district_polygon(local, width * 1000.0, height * 1000.0, rotation)
        try:
            ts_params = SpatialModelParams(
                model=ts_model, count=int(count),
                vertices_min=int(vertices_min), vertices_max=int(vertices_max),
                area_min_m2=float(area_min), area_max_m2=float(area_max),
                irregularity=float(irregularity), max_aspect_ratio=float(max_aspect),
                rotation_min_deg=float(rot_min), rotation_max_deg=float(rot_max),
                boundary_margin_m=float(boundary_margin), min_distance_m=float(min_distance),
                n_clusters=int(ts_clusters), cluster_sigma_m=float(ts_sigma),
                jitter_fraction=float(ts_jitter), n_lines=int(ts_lines),
                line_sigma_m=400.0,
            )
        except ValueError as exc:
            st.error(f"Ошибка параметров: {exc}")
            st.stop()

        progress = st.progress(0.0, text="Статистические испытания…")
        study = run_target_model_study(
            ts_params, district, realizations=int(ts_n), seed_start=int(ts_seed),
            beta=float(ts_beta), grid=5, models=MODELS,
            progress=lambda d, t: progress.progress(d / (3 * t),
                                                    text=f"Испытания: {d}/{t}"),
        )
        nn_emp = collect_nn_samples(
            ts_params, district, models=MODELS, realizations=int(ts_n),
            seed_start=int(ts_seed),
            progress=lambda d, t: progress.progress(1 / 3 + d / (3 * t),
                                                    text=f"Выборки F*(r): {d}/{t}"),
        )
        nn_ref = collect_nn_samples(
            ts_params, district, models=MODELS, realizations=int(ts_n),
            seed_start=int(ts_seed) + 100_000,
            progress=lambda d, t: progress.progress(2 / 3 + d / (3 * t),
                                                    text=f"Имитационный эталон: {d}/{t}"),
        )
        progress.empty()

        examples = {}
        for m in MODELS:
            p = SpatialModelParams(**study["per_model"][m]["params"])
            examples[m] = generate_target_objects(
                p, district, np.random.default_rng(np.random.SeedSequence([int(ts_seed), 20261126, len(m)])))

        out_dir = tempfile.mkdtemp(prefix="sro_target_stat_")
        charts = render_statistics_charts(
            study, examples, nn_emp, nn_ref,
            selected_model=ts_model, metric=ts_metric,
            beta=float(ts_beta), epsilon0=float(ts_eps0),
            out_dir=out_dir, formats=("png", "svg"),
        )
        st.session_state["ts_study"] = study
        st.session_state["ts_charts"] = charts
        st.session_state["ts_settings"] = {
            "n": int(ts_n), "beta": float(ts_beta), "eps0": float(ts_eps0),
            "seed": int(ts_seed), "model": ts_model, "metric": ts_metric,
            "params": ts_params,
        }

    ts_study = st.session_state.get("ts_study")
    if ts_study is not None:
        ts_settings = st.session_state["ts_settings"]
        ts_charts = st.session_state["ts_charts"]
        p_used: SpatialModelParams = ts_settings["params"]

        st.markdown("**Таблица 1. Исходные параметры математического моделирования**")
        st.dataframe(pd.DataFrame([
            {"Параметр": "Размер района, км", "Значение": f"{width:.1f} × {height:.1f}"},
            {"Параметр": "Количество объектов", "Значение": p_used.count},
            {"Параметр": "Площади объектов, м²", "Значение": f"{p_used.area_min_m2:,.0f} – {p_used.area_max_m2:,.0f}"},
            {"Параметр": "Вершин контура", "Значение": f"{p_used.vertices_min} – {p_used.vertices_max}"},
            {"Параметр": "Законы распределения", "Значение": "равномерно-случайный, кластерный, регулярный, линейный"},
            {"Параметр": "Параметры кластерного закона", "Значение": f"K = {p_used.n_clusters}, σ = {p_used.cluster_sigma_m:,.0f} м"},
            {"Параметр": "Параметр регулярного закона", "Значение": f"дрожание j = {p_used.jitter_fraction:.2f}"},
            {"Параметр": "Параметры линейного закона", "Значение": f"L = {p_used.n_lines}, σ_l = {p_used.line_sigma_m:,.0f} м"},
            {"Параметр": "Ограничения размещения", "Значение": f"отступ {p_used.boundary_margin_m:,.0f} м; мин. расстояние {p_used.min_distance_m:,.0f} м; пересечения запрещены"},
            {"Параметр": "Объём испытаний n", "Значение": ts_settings["n"]},
            {"Параметр": "Доверительная вероятность β", "Значение": ts_settings["beta"]},
            {"Параметр": "Заданная точность ε₀, м", "Значение": ts_settings["eps0"]},
            {"Параметр": "Начальный seed", "Значение": ts_settings["seed"]},
        ]))

        metric_t = ts_settings["metric"]
        st.markdown(f"**Таблица 2. Результаты статистического оценивания** "
                    f"(характеристика: {ts_study['metric_titles_ru'][metric_t]}, "
                    f"n = {ts_settings['n']}, β = {ts_settings['beta']:.2f})")
        rows = []
        for m in MODELS:
            pm = ts_study["per_model"][m]
            last = pm["classical"][metric_t][-1]
            skew = sample_skewness([r[metric_t] for r in pm["runs"]])
            rows.append({
                "Закон": MODEL_TITLES_RU[m],
                "M*(n)": round(last["m_star"], 4),
                "D̃(n)": round(last["d_tilde"], 6),
                "σ̃(n)": round(last["sigma_tilde"], 4),
                "ε(β,n)": round(last["epsilon"], 4),
                "I нижн.": round(last["ci_low"], 4),
                "I верхн.": round(last["ci_high"], 4),
                "Асимметрия g₁": round(skew, 3),
                "Отклон. CSR (z)": pm["ce_z_reject_share"],
                "Отклон. CSR (χ²)": pm["csr_chi2_reject_share"],
            })
        st.dataframe(pd.DataFrame(rows))
        st.caption(
            "M*(n) = Σxᵢ/n; D̃(n) = Σ(xᵢ−M*)²/(n−1); σ̃ = √D̃; ε = t[(1+β)/2; n−1]·σ̃/√n; "
            "I = [M*−ε; M*+ε]. xᵢ — пространственная характеристика независимой реализации. "
            "Асимметрия g₁ — индикатор применимости t-интервалов: при |g₁| ≲ 0,5 и n ≥ 30 "
            "интервалы устойчивы; точные χ²-интервалы дисперсии без обоснования нормальности "
            "не применяются."
        )

        st.markdown("**Шесть обязательных научных графиков**")
        for name in sorted(ts_charts):
            png = next(p for p in ts_charts[name] if p.suffix == ".png")
            st.image(str(png))

        st.subheader("Экспорт графиков")
        import zipfile

        buf_png = io.BytesIO()
        with zipfile.ZipFile(buf_png, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, ps in ts_charts.items():
                for p in ps:
                    zf.write(p, arcname=p.name)
        st.download_button("Скачать все графики (PNG + SVG, zip)", buf_png.getvalue(),
                           file_name="target_statistics_charts.zip", mime="application/zip")

# ---------------------------------------------------------------------------
# Вкладка 5. Совместная модель района ДЗЗ
# ---------------------------------------------------------------------------
with tab_joint:
    st.subheader("Совместная стохастическая модель района ДЗЗ: целевые объекты T "
                 "и препятствия оптической съёмке B")
    st.caption(
        "Объединение двух пространственных моделей на одном районе (панель «Район» слева) "
        "с независимыми seed: объекты T — четыре закона размещения (как во вкладке «Модель "
        "объектов съёмки»), препятствия B — поля облачности тем же ядром (пересечения полей "
        "разрешены). Геометрия досъёмки: U = T ∩ B — участки, требующие РСА; "
        "P_опт = S(T \\ B)/S(T) — исходная полнота оптического покрытия; "
        "P_скр = S(T ∩ B)/S(T) — доля площади объектов, требующая досъёмки; "
        "инвариант S(U) + S(T \\ B) = S(T) контролируется."
    )

    jc1, jc2, jc3 = st.columns(3)
    j_law = jc1.selectbox("Закон объектов T", list(MODELS),
                          format_func=lambda m: MODEL_TITLES_RU[m], key="j_law")
    j_seed_t = jc2.number_input("Seed объектов", value=42, step=1, key="j_seed_t")
    j_seed_b = jc3.number_input("Seed препятствий", value=7, step=1, key="j_seed_b")

    jc4, jc5, jc6 = st.columns(3)
    j_n_clusters = jc4.number_input("Групп (кластерный T)", value=5, min_value=1, step=1,
                                    disabled=j_law != "cluster", key="j_nc")
    j_sigma = jc5.number_input("СКО группы T, м", value=1500.0, min_value=1.0,
                               disabled=j_law != "cluster", key="j_sig")
    j_jitter = jc6.slider("Дрожание решётки T", 0.0, 1.0, 0.6,
                          disabled=j_law != "regular", key="j_jit")
    jc7, jc8, _ = st.columns(3)
    j_n_lines = jc7.number_input("Линейных структур T", value=4, min_value=1, step=1,
                                 disabled=j_law != "linear", key="j_nl")

    st.markdown("**Препятствия B (поля облачности)**")
    jb1, jb2, jb3, jb4 = st.columns(4)
    j_use_obstacles = jb1.checkbox("Моделировать препятствия", value=True, key="j_use_b")
    j_b_law = jb2.selectbox("Закон препятствий", list(MODELS), index=0,
                            format_func=lambda m: MODEL_TITLES_RU[m],
                            disabled=not j_use_obstacles, key="j_b_law")
    j_b_count = jb3.number_input("Число полей B", value=8, min_value=1, step=1,
                                 disabled=not j_use_obstacles, key="j_b_count")
    j_b_area_min = jb4.number_input("Площадь поля мин, км²", value=20.0, min_value=0.1,
                                    disabled=not j_use_obstacles, key="j_b_amin")
    j_b_area_max = st.number_input("Площадь поля макс, км²", value=60.0, min_value=0.1,
                                   disabled=not j_use_obstacles, key="j_b_amax")

    st.markdown("**Слои карты**")
    jl1, jl2, jl3, jl4, jl5 = st.columns(5)
    j_l_obst = jl1.checkbox("Препятствия", value=True, key="j_l_b")
    j_l_obj = jl2.checkbox("Контуры объектов", value=True, key="j_l_o")
    j_l_vis = jl3.checkbox("Видимая часть", value=True, key="j_l_v")
    j_l_resh = jl4.checkbox("Досъёмка U", value=True, key="j_l_r")
    j_l_dist = jl5.checkbox("Район", value=True, key="j_l_d")

    # Слои карты только меняют отрисовку. Изменение любого расчётного параметра
    # обязано инвалидировать прежний результат, чтобы не показывать чужую
    # геометрию (например, 25 объектов после смены count на 150).
    j_controls = (
        float(center_lat), float(center_lon), float(width), float(height), float(rotation),
        int(count), int(vertices_min), int(vertices_max), float(area_min), float(area_max),
        float(irregularity), float(rot_min), float(rot_max), float(max_aspect),
        float(min_distance), float(boundary_margin), int(max_attempts),
        str(j_law), int(j_seed_t), int(j_seed_b), int(j_n_clusters), float(j_sigma),
        float(j_jitter), int(j_n_lines), bool(j_use_obstacles), str(j_b_law),
        int(j_b_count), float(j_b_area_min), float(j_b_area_max),
    )
    if st.session_state.get("j_controls") != j_controls:
        if st.session_state.get("j_result") is not None:
            st.info("Параметры совместной модели изменились. Нажмите «Построить совместную модель».")
        st.session_state["j_result"] = None
        st.session_state["j_summary"] = None
        st.session_state["j_local"] = None
        st.session_state["j_controls"] = j_controls

    if st.button("Построить совместную модель", type="primary", key="j_run"):
        # Не оставляем прежний результат при ошибке параметров/размещения.
        st.session_state["j_result"] = None
        st.session_state["j_summary"] = None
        st.session_state["j_local"] = None
        local = LocalCRS(center_lat, center_lon)
        district = build_district_polygon(local, width * 1000.0, height * 1000.0, rotation)
        try:
            j_t_params = SpatialModelParams(
                model=j_law, count=int(count),
                vertices_min=int(vertices_min), vertices_max=int(vertices_max),
                area_min_m2=float(area_min), area_max_m2=float(area_max),
                irregularity=float(irregularity), max_aspect_ratio=float(max_aspect),
                rotation_min_deg=float(rot_min), rotation_max_deg=float(rot_max),
                boundary_margin_m=float(boundary_margin), min_distance_m=float(min_distance),
                max_attempts_per_object=int(max_attempts),
                n_clusters=int(j_n_clusters), cluster_sigma_m=float(j_sigma),
                jitter_fraction=float(j_jitter), n_lines=int(j_n_lines), line_sigma_m=400.0,
            )
            j_b_params = None
            if j_use_obstacles:
                if j_b_area_min > j_b_area_max:
                    st.error("Площадь поля мин не может превышать макс.")
                    st.stop()
                j_b_params = SpatialModelParams(
                    model=j_b_law, count=int(j_b_count),
                    vertices_min=6, vertices_max=10,
                    area_min_m2=float(j_b_area_min) * 1e6,
                    area_max_m2=float(j_b_area_max) * 1e6,
                    irregularity=0.5, max_aspect_ratio=2.5,
                    boundary_margin_m=0.0, allow_overlap=True,
                    n_clusters=min(3, int(j_b_count)), cluster_sigma_m=3000.0,
                    jitter_fraction=0.6, n_lines=3, line_sigma_m=1500.0,
                )
        except ValueError as exc:
            st.error(f"Ошибка параметров: {exc}")
            st.stop()
        with st.spinner("Генерация совместной модели и геометрическое пересечение…"):
            try:
                j_result = generate_joint_model(
                    j_t_params, j_b_params, district,
                    target_seed=int(j_seed_t), obstacle_seed=int(j_seed_b))
            except PlacementError as exc:
                st.error(f"Ошибка размещения: {exc}")
                st.stop()
            j_summary = joint_summary(j_result)
        st.session_state["j_result"] = j_result
        st.session_state["j_summary"] = j_summary
        st.session_state["j_local"] = local

    j_result = st.session_state.get("j_result")
    if j_result is not None:
        j_summary = st.session_state["j_summary"]
        j_map = render_joint_map(
            j_result, "/tmp/sro_joint_map.png",
            layers={"district": j_l_dist, "obstacles": j_l_obst, "objects": j_l_obj,
                    "visible": j_l_vis, "reshoot": j_l_resh})
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.image(str(j_map), caption="Совместная модель района ДЗЗ")
        with col_r:
            st.subheader("Геометрическая полнота")
            st.metric("Сгенерировано объектов T", f"{j_summary['n_targets']}")
            st.metric("P_опт — оптическое покрытие объектов", f"{j_summary['p_opt']:.4f}")
            st.metric("P_скр — доля, требующая досъёмки РСА", f"{j_summary['p_skr']:.4f}")
            st.metric("S(T), км²", f"{j_summary['area_targets_m2'] / 1e6:.2f}")
            st.metric("S(U) = S(T ∩ B), км²", f"{j_summary['area_reshoot_m2'] / 1e6:.2f}")
            st.metric("S(T \\ B), км²", f"{j_summary['area_visible_m2'] / 1e6:.2f}")
            c = j_summary["counts"]
            st.metric("Объекты: видимые / частичные / скрытые",
                      f"{c['visible']} / {c['partial']} / {c['hidden']}")
            st.caption(
                f"Инвариант P_опт + P_скр = 1: "
                f"{'выполнен' if j_summary['area_identity_ok'] else 'НАРУШЕН'}. "
                f"Препятствий B: {j_summary['n_obstacles']} "
                f"(seed {j_summary['obstacle_seed']}); объекты T — seed "
                f"{j_summary['target_seed']} (независимы)."
            )

        st.subheader("Геометрические характеристики объектов")
        st.dataframe(pd.DataFrame([
            {
                "object_id": r["object_id"],
                "area_m2": round(r["area_m2"], 1),
                "hidden_area_m2": round(r["hidden_area_m2"], 1),
                "hidden_fraction": round(r["hidden_fraction"], 4),
                "статус": {"visible": "видим", "partial": "частично скрыт",
                           "hidden": "скрыт"}[r["status"]],
            }
            for r in j_summary["per_object"]
        ]))

        import json as _json2

        j_layers = export_joint_geojson(j_result, st.session_state["j_local"])
        jd1, jd2, jd3 = st.columns(3)
        buf_j = io.BytesIO()
        import zipfile as _zipfile

        with _zipfile.ZipFile(buf_j, "w", _zipfile.ZIP_DEFLATED) as zf:
            for name, fc in j_layers.items():
                zf.writestr(f"joint_{name}.geojson",
                            _json2.dumps(fc, ensure_ascii=False))
        jd1.download_button("Скачать GeoJSON-слои (zip)", buf_j.getvalue(),
                            file_name="joint_model_geojson.zip", mime="application/zip")
        jd2.download_button("Скачать сводку (JSON)",
                            _json2.dumps({k: v for k, v in j_summary.items()},
                                         ensure_ascii=False, indent=2),
                            file_name="joint_model_summary.json", mime="application/json")
        with open(j_map, "rb") as f:
            jd3.download_button("Скачать карту (PNG)", f.read(),
                                file_name="joint_model_map.png", mime="image/png")
        st.caption(
            "GeoJSON (RFC 7946, WGS 84): слои district (граница района), objects "
            "(объекты T со статусом и долей скрытой площади), obstacles (препятствия B), "
            "reshoot (полигоны участков U = T ∩ B для планирования РСА-досъёмки в plane_sar), "
            "visible (видимая оптике часть T \\ B)."
        )
