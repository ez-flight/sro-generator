"""Тесты обратной совместимости 1.0 → 1.1: конфигурации, CLI, экспорт."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import sro_generator
from sro_generator.cli import main as cli_main
from sro_generator.config import ScenarioConfig
from sro_generator.export import export_scenario
from sro_generator.generator import generate_scenario
from sro_generator.placement import PlacementError

ROOT = Path(__file__).resolve().parent.parent


def make_config(**overrides) -> ScenarioConfig:
    data = json.loads((ROOT / "example_config.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        parts = key.split("__")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return ScenarioConfig.model_validate(data)


class TestVersion:
    def test_version_is_2_7_0(self):
        assert sro_generator.__version__ == "2.7.0"


class TestLegacyConfig:
    def test_1_0_json_loads_with_defaults(self):
        """JSON конфигурации 1.0 (без новых полей) валиден; новые поля по умолчанию."""
        cfg = ScenarioConfig.from_json(ROOT / "example_config.json")
        assert cfg.placement.center_region_mode == "exact"
        assert cfg.placement.spatial_index is True

    def test_1_0_json_generates(self):
        cfg = ScenarioConfig.from_json(ROOT / "example_config.json")
        scenario = generate_scenario(cfg)
        assert len(scenario.objects) == cfg.objects.count
        assert len(scenario.frames) == cfg.objects.count * len(cfg.frames.pass_azimuths_deg)

    def test_legacy_region_mode_from_config(self):
        cfg = make_config(placement__center_region_mode="legacy")
        scenario = generate_scenario(cfg)
        assert len(scenario.objects) == cfg.objects.count


class TestCli:
    def test_cli_generate_ok(self, tmp_path):
        code = cli_main(["generate", "--config", str(ROOT / "example_config.json"),
                         "--output", str(tmp_path)])
        assert code == 0
        for name in ("district.geojson", "objects.geojson", "objects.csv", "frames.geojson",
                     "config.resolved.json", "generation_report.json", "preview.png"):
            assert (tmp_path / name).exists(), name

    def test_cli_report_contains_version_and_counters(self, tmp_path):
        """Счётчики проверок > 0 на плотном сценарии (в разреженной демо-конфигурации
        STRtree и фильтр по ограничивающим прямоугольникам могут законно дать 0
        точных проверок — совпадение с brute force контролирует тест эквивалентности)."""
        cfg = make_config(district__width_km=40.0, district__height_km=40.0, objects__count=120)
        cfg_path = tmp_path / "dense.json"
        cfg_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        out_dir = tmp_path / "out"
        cli_main(["generate", "--config", str(cfg_path), "--output", str(out_dir)])
        report = json.loads((out_dir / "generation_report.json").read_text(encoding="utf-8"))
        assert report["version"] == "2.7.0"
        assert report["placement_exact_checks_total"] > 0
        assert report["placement_tree_queries"] > 0
        assert report["checks"]["count_matches_config"]
        assert report["checks"]["all_polygons_valid"]
        assert report["checks"]["areas_within_range"]

    def test_cli_bad_config_exit_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"seed": 1}), encoding="utf-8")
        code = cli_main(["generate", "--config", str(bad), "--output", str(tmp_path / "o")])
        assert code == 2
        assert "Ошибка конфигурации" in capsys.readouterr().err

    def test_cli_missing_file_exit_2(self, tmp_path, capsys):
        code = cli_main(["generate", "--config", str(tmp_path / "nope.json"), "--output", str(tmp_path / "o")])
        assert code == 2
        assert "не найден" in capsys.readouterr().err

    def test_cli_placement_failure_exit_1(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.json"
        cfg = make_config(
            district__width_km=5.0, district__height_km=5.0,
            objects__count=200, objects__area_max_m2=5_000_000.0,
            placement__max_attempts_per_object=20,
        )
        cfg_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        code = cli_main(["generate", "--config", str(cfg_path), "--output", str(tmp_path / "o")])
        assert code == 1
        assert "Ошибка размещения" in capsys.readouterr().err

    def test_cli_module_invocation(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "sro_generator.cli", "generate",
             "--config", str(ROOT / "example_config.json"), "--output", str(tmp_path)],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode == 0
        assert "Готово" in result.stdout


class TestExportRoundtrip:
    def test_export_files_readable(self, tmp_path):
        from shapely.geometry import shape

        scenario = generate_scenario(make_config(objects__count=7))
        paths = export_scenario(scenario, tmp_path)
        objs = json.loads(paths["objects.geojson"].read_text(encoding="utf-8"))
        assert len(objs["features"]) == 7
        for feat in objs["features"]:
            geom = shape(feat["geometry"])
            assert geom.is_valid
        csv_text = paths["objects.csv"].read_text(encoding="utf-8")
        assert "object_id" in csv_text.splitlines()[0]
        # preview.png непуст
        assert paths["preview.png"].stat().st_size > 0

    def test_wgs84_coordinates(self, tmp_path):
        from shapely.geometry import shape

        scenario = generate_scenario(make_config())
        paths = export_scenario(scenario, tmp_path)
        district = json.loads(paths["district.geojson"].read_text(encoding="utf-8"))
        geom = shape(district["features"][0]["geometry"])
        lons = [c[0] for c in geom.exterior.coords]
        lats = [c[1] for c in geom.exterior.coords]
        assert all(-180 <= lo <= 180 for lo in lons)
        assert all(-90 <= la <= 90 for la in lats)
        # район 100×90 км вокруг (60E, 55N)
        assert 59.0 < min(lons) and max(lons) < 61.2
        assert 54.5 < min(lats) and max(lats) < 55.6


class TestStreamlitApp:
    def test_app_compiles(self):
        """Streamlit-приложение синтаксически корректно (полный запуск — вручную: streamlit run app.py)."""
        import py_compile

        py_compile.compile(str(ROOT / "app.py"), doraise=True)
