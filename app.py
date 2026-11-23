"""Графический интерфейс генератора района СРО (Streamlit).

Одиночный прогон: кнопка «Сгенерировать».
Серия прогонов: задайте число прогонов и потоков — прогоны выполняются
параллельно (seed = базовый … базовый + N − 1) и сводятся в статистику.
Исследование сходимости: M независимых экспериментов × K итераций
итерационного алгоритма формирования кадров на фиксированном сценарии.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile

import pandas as pd
import streamlit as st

from sro_generator.config import ConfigError, ScenarioConfig
from sro_generator.convergence import (
    ConvergenceParams,
    render_convergence_charts,
    run_convergence_study,
    write_convergence_csv,
    write_convergence_json,
)
from sro_generator.export import render_preview
from sro_generator.generator import generate_scenario
from sro_generator.placement import PlacementError
from sro_generator.stats import (
    render_analytics_charts,
    render_report_charts,
    run_batch,
    write_batch_csv,
)

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
    seed = st.number_input("Seed (базовый)", value=42, step=1)

    st.header("Серия прогонов")
    n_runs = st.number_input("Число прогонов", value=10, min_value=2, max_value=2000, step=10)
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
    """Собрать конфигурацию из параметров сайдбара (та же структура, что и раньше)."""
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


def render_batch(batch: dict, out_dir: str) -> None:
    """Показать результаты серии: метрики, графики, таблицы, выгрузки."""
    s = batch["summary"]
    st.success(f"Серия завершена: успешно {s['n_ok']} из {s['n_runs']} прогонов.")
    if s["n_failed"]:
        st.warning(f"Упавших прогонов: {s['n_failed']} (см. вкладку «Ошибки»).")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Попыток размещения (среднее)", f"{s['attempts_total']['mean']:,.0f}")
    m2.metric("Время прогона, с (среднее)", f"{s['duration_s']['mean']:.3f}")
    m3.metric("Fill ratio (среднее)", f"{s['fill_ratio']['mean']:.4f}")
    m4.metric("Покрытие объектов (среднее)", f"{s['coverage_fraction']['mean']:.3f}")
    m5.metric("Доля полных кадров", f"{s['full_coverage_share']:.1%}")

    tab_charts, tab_analytics, tab_runs, tab_files, tab_errors = st.tabs(
        ["Графики", "Аналитика", "Прогоны", "CSV", "Ошибки"]
    )

    chart_paths = render_report_charts(batch, out_dir)
    csv_paths = write_batch_csv(batch, out_dir)

    with tab_charts:
        for p in chart_paths.values():
            st.image(str(p))

    with tab_analytics:
        st.caption(
            "Статистические проверки — это проверки согласия с заявленной гипотезой, "
            "а не доказательство равномерности: ограничения размещения систематически "
            "меняют распределение принимаемых объектов."
        )
        tests = s.get("statistical_tests", {})
        if tests:
            st.dataframe(pd.DataFrame([
                {
                    "тест": name,
                    "тип": t.get("kind"),
                    "статистика": (round(t["statistic"], 6)
                                   if isinstance(t.get("statistic"), float) else None),
                    "df": t.get("df"),
                    "p-value": (round(t["p_value"], 6)
                                if isinstance(t.get("p_value"), float) else None),
                    "n": t.get("n"),
                    "применим": t.get("applicable"),
                    "предупреждения": " | ".join(t.get("warnings", [])),
                }
                for name, t in tests.items()
            ]))
        if s.get("clark_evans_warnings"):
            for w in s["clark_evans_warnings"]:
                st.warning(w)
        analytics_paths = render_analytics_charts(batch, out_dir)
        for p in analytics_paths.values():
            st.image(str(p))

    with tab_runs:
        st.dataframe(pd.DataFrame([
            {
                "seed": r["seed"],
                "ok": r["ok"],
                "duration_s": round(r.get("duration_s", 0.0), 4),
                "attempts_total": r.get("attempts_total"),
                "fill_ratio": round(r.get("fill_ratio", 0.0), 6),
                "nnd_mean_m": round(r.get("nnd_mean_m", 0.0), 1),
                "n_objects": len(r.get("objects", [])),
                "frames_total": len(r.get("coverage", [])),
                "full_coverage_frames": r.get("full_coverage_frames"),
                "warnings": len(r.get("warnings", [])),
            }
            for r in batch["runs"]
        ]))
        st.caption(
            "χ²/df: вершины = %.3g, категории = %.3g, ячейки района = %.3g "
            "(значения около 1.0 — распределение близко к равномерному)"
            % (s["vertex_chi2_df"], s["class_chi2_df"], s["quadrant_chi2_df"])
        )

    with tab_files:
        for name_csv, p in csv_paths.items():
            with open(p, "rb") as f:
                st.download_button(f"Скачать {name_csv}", f.read(), file_name=name_csv, mime="text/csv")
        st.caption(
            "runs_summary.csv — метрики по каждому прогону; "
            "objects_all.csv — все объекты всех прогонов (для pandas/R: сводные таблицы, критерии согласия, корреляции)."
        )

    with tab_errors:
        if s["errors"]:
            for e in s["errors"]:
                st.error(f"seed={e['seed']}: {e['error']}")
        else:
            st.info("Ошибок нет — все прогоны завершились успешно.")


_VERDICT_TO_MESSAGE = {
    "converged": st.success,
    "stagnation": st.warning,
    "different_local_optima": st.warning,
    "no_convergence": st.error,
}


def render_convergence(study: dict, out_dir: str) -> None:
    """Показать результаты исследования сходимости: диагноз, метрики, графики, таблицы, выгрузки."""
    c = study["convergence"]
    params = study["params"]
    _VERDICT_TO_MESSAGE[c["verdict"]](c["verdict_ru"] + ".")
    st.caption(c["detail"])
    for w in study.get("warnings", []):
        st.warning(w)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Итоговая средняя полнота", f"{c['final_mean']:.4f}")
    m2.metric("Итоговое std между экспериментами", f"{c['final_std']:.4f}")
    m3.metric("Разброс p10–p90 (на K)", f"{c['final_band_p10_p90']:.4f}")
    m4.metric("Стабилизировалось",
              f"на k = {c['stabilized_at_iteration']}" if c["stabilized_at_iteration"] else "нет")
    m5.metric("Экспериментов × итераций", f"{params['experiments']} × {params['iterations']}")

    tab_summary, tab_exp, tab_files = st.tabs(["Сводка по итерациям", "Эксперименты", "CSV / JSON / PNG"])

    chart_paths = render_convergence_charts(study, out_dir)
    csv_paths = write_convergence_csv(study, out_dir)
    json_path = write_convergence_json(study, out_dir)

    with tab_summary:
        st.dataframe(pd.DataFrame([
            {
                "k": s["iteration"],
                "среднее": round(s["mean"], 6),
                "std": round(s["std"], 6),
                "медиана": round(s["median"], 6),
                "p10": round(s["p10"], 6),
                "p90": round(s["p90"], 6),
                "ДИ95 нижн.": round(s["ci95_low"], 6),
                "ДИ95 верхн.": round(s["ci95_high"], 6),
                "средний прирост": round(s["mean_increment"], 6),
            }
            for s in study["iteration_stats"]
        ]))
        st.caption(
            "Статистики по фактическим результатам M независимых экспериментов на каждой итерации; "
            "std — ddof=1, доверительный интервал среднего — 95%-й (t-критическое). "
            "Полнота I(k) = 0.5·(покрытие района ∪ кадров + покрытие объектов ∪ кадров), "
            "перекрытия в объединении не считаются дважды."
        )
        for p in chart_paths.values():
            st.image(str(p))

    with tab_exp:
        st.dataframe(pd.DataFrame([
            {
                "эксперимент": e["experiment"],
                "seed": e["seed"],
                "итоговая полнота": round(e["final_completeness"], 6),
                "время, с": round(e["duration_s"], 3),
                "выбрано кадров": sum(1 for f in e["frames_selected"] if f),
            }
            for e in study["experiments"]
        ]))
        st.caption("Полная траектория I(k) каждого эксперимента — в convergence_trajectories.csv и convergence.json.")

    with tab_files:
        for name_csv, p in csv_paths.items():
            with open(p, "rb") as f:
                st.download_button(f"Скачать {name_csv}", f.read(), file_name=name_csv, mime="text/csv")
        with open(json_path, "rb") as f:
            st.download_button("Скачать convergence.json", f.read(), file_name="convergence.json",
                               mime="application/json")
        for name_png, p in chart_paths.items():
            with open(p, "rb") as f:
                st.download_button(f"Скачать {name_png}", f.read(), file_name=name_png, mime="image/png")
        st.caption(
            "convergence_trajectories.csv — полная траектория I(k) каждого эксперимента; "
            "convergence_summary.csv — статистика по номерам итераций; "
            "convergence_experiments.csv — итоги экспериментов; "
            "convergence.json — всё исследование (параметры, траектории, диагноз)."
        )


tab_gen, tab_conv = st.tabs(["Генерация и серия", "Сходимость алгоритма"])

with tab_gen:
    st.subheader("Ручные центры (пусто — полностью случайное размещение)")
    centers_df = st.data_editor(
        pd.DataFrame(columns=["object_id", "latitude", "longitude"]),
        num_rows="dynamic",
        key="manual_centers",
    )

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

    st.divider()
    st.subheader("Серия прогонов для анализа и статистики")
    st.caption(
        f"Каждый прогон — независимая генерация со своим seed "
        f"({int(seed)} … {int(seed) + int(n_runs) - 1}). Прогоны идут параллельно в {int(workers)} поток(ах); "
        "упавшие прогоны фиксируются и не прерывают серию."
    )

    if st.button(f"Запустить серию: {int(n_runs)} прогонов", type="primary"):
        manual_centers, errors = parse_manual_centers(centers_df)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
        config_dict = build_config(manual_centers)
        try:
            ScenarioConfig.model_validate(config_dict)
        except ConfigError as exc:
            st.error(f"Ошибка конфигурации: {exc}")
            st.stop()

        progress = st.progress(0.0, text="Серия прогонов…")
        batch = run_batch(
            config_dict,
            n_runs=int(n_runs),
            workers=int(workers),
            seed_start=int(seed),
            progress=lambda done, total: progress.progress(done / total, text=f"Прогонов завершено: {done}/{total}"),
        )
        progress.empty()
        st.session_state["batch"] = batch
        st.session_state["batch_dir"] = tempfile.mkdtemp(prefix="sro_batch_")

    batch = st.session_state.get("batch")
    if batch:
        render_batch(batch, st.session_state.get("batch_dir") or tempfile.mkdtemp(prefix="sro_batch_"))

with tab_conv:
    st.subheader("Исследование сходимости итерационного алгоритма формирования кадров РСА")
    st.caption(
        "Для фиксированного сценария (район, объекты и кадры-кандидаты неизменны — "
        "случайность только в алгоритме) проводится M независимых экспериментов, "
        "каждый содержит K последовательных итераций рандомизированного жадного отбора кадров. "
        "После каждой итерации сохраняется полнота информации I(k) — покрытие района и объектов "
        "объединением выбранных кадров без повторного счёта перекрытий."
    )
    cc1, cc2, cc3 = st.columns(3)
    conv_m = cc1.number_input("Экспериментов M", value=20, min_value=2, max_value=500, step=5)
    conv_k = cc2.number_input("Итераций K", value=30, min_value=1, max_value=500, step=5)
    conv_seed = cc3.number_input("Начальный seed экспериментов", value=0, step=1)
    cc4, cc5, cc6 = st.columns(3)
    conv_tol = cc4.number_input("Критерий стабилизации: порог прироста", value=0.001, min_value=0.000001,
                                format="%.6f")
    conv_window = cc5.number_input("Критерий стабилизации: окно, итераций", value=3, min_value=1, step=1)
    conv_min = cc6.number_input("Минимально требуемая полнота", value=0.9, min_value=0.0, max_value=1.0)
    cc7, cc8 = st.columns(2)
    conv_temp = cc7.number_input("Температура алгоритма", value=0.25, min_value=0.001, format="%.3f")
    conv_workers = cc8.number_input(
        "Параллельных потоков (исследование)",
        value=min(4, os.cpu_count() or 1),
        min_value=1, max_value=32, step=1,
        key="conv_workers",
    )

    if st.button("Запустить исследование сходимости", type="primary"):
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
        if not config.frames.enabled:
            st.error("Для исследования сходимости включите кадры РСА (панель «Кадры РСА» слева).")
            st.stop()
        n_candidates = config.objects.count * len(config.frames.pass_azimuths_deg)
        if conv_k > n_candidates:
            st.info(
                f"Кадров-кандидатов {n_candidates} меньше K = {int(conv_k)}: после исчерпания кадров "
                "полнота перестанет расти, а итерации досчитаются константой."
            )
        try:
            params = ConvergenceParams(
                experiments=int(conv_m),
                iterations=int(conv_k),
                seed_start=int(conv_seed),
                stabilization_tol=float(conv_tol),
                stabilization_window=int(conv_window),
                min_completeness=float(conv_min),
                temperature=float(conv_temp),
            )
        except ValueError as exc:
            st.error(f"Ошибка параметров исследования: {exc}")
            st.stop()

        progress = st.progress(0.0, text="Исследование сходимости…")
        study = run_convergence_study(
            config,
            params,
            workers=int(conv_workers),
            progress=lambda done, total: progress.progress(done / total, text=f"Экспериментов завершено: {done}/{total}"),
        )
        progress.empty()
        st.session_state["convergence"] = study
        st.session_state["convergence_dir"] = tempfile.mkdtemp(prefix="sro_conv_")

    study = st.session_state.get("convergence")
    if study:
        render_convergence(study, st.session_state.get("convergence_dir") or tempfile.mkdtemp(prefix="sro_conv_"))
