"""CLI пакетной генерации: python -m sro_generator.cli generate --config C --output O."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, ScenarioConfig
from .export import export_scenario
from .generator import generate_scenario
from .placement import PlacementError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sro_generator", description="Генератор района СРО")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Сгенерировать район по JSON-конфигурации")
    gen.add_argument("--config", required=True, help="Путь к JSON-конфигурации (UTF-8)")
    gen.add_argument("--output", required=True, help="Каталог для файлов результата")
    args = parser.parse_args(argv)

    if args.command == "generate":
        try:
            config = ScenarioConfig.from_json(args.config)
        except ConfigError as exc:
            print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError:
            print(f"Ошибка: файл конфигурации не найден: {args.config}", file=sys.stderr)
            return 2
        try:
            scenario = generate_scenario(config)
            paths = export_scenario(scenario, args.output)
        except PlacementError as exc:
            print(f"Ошибка размещения: {exc}", file=sys.stderr)
            return 1
        print(f"Готово: создано объектов {len(scenario.objects)}, кадров {len(scenario.frames)}.")
        for name, p in sorted(paths.items()):
            print(f"  {name} -> {p}")
        for w in scenario.warnings:
            print(f"Предупреждение: {w}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
