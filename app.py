"""Графический интерфейс SRO Generator 2.0.0 (Streamlit).

Две вкладки:
1. «Генерация СРО» — настройка района и объектов, воспроизводимая генерация
   (seed), визуализация района, объектов, облачности и кадров досъёмки РСА,
   расчёт геометрической полноты покрытия (G — район, C — скрытые объекты).
2. «Статистическое исследование» — оценивание математического ожидания и
   дисперсии геометрической полноты покрытия района методом статистических
   испытаний (M*(n), D̃(n), σ̃(n), ε(β,n), пять обязательных графиков).

Исследовательские стенды — в benchmarks/ (convergence_analysis.py,
monte_carlo_convergence.py, cloud_coverage_analysis.py); вычислительные
функции сохранены в sro_generator/convergence.py и sro_generator/monte_carlo.py.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.generator import generate_scenario
from sro_generator.monte_carlo import (
    classical_series,
    k_sweep,
    render_classical_charts,
    required_trials,
    run_monte_carlo_experiment,
    MonteCarloParams,
)
from sro_generator.placement import PlacementError

st.set_page_config(page_title="SRO Generator 2.0", layout="wide")
st.title("SRO Generator 2.0 — генерация СРО и статистическое исследование досъёмки РСА")

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


tab_gen, tab_stat = st.tabs(["Генерация СРО", "Статистическое исследование"])

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
