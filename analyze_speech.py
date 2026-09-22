#!/usr/bin/env python3
"""Разбор речи по расшифровке — обёртка над пакетом speech_analysis."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_analysis import analyze, config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--words", type=Path, help="пословные таймкоды (ищутся рядом)")
    ap.add_argument("--names", help="имена вместо SPEAKER_00, через запятую")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--show-config", action="store_true",
                    help="показать путь и содержимое настроек")
    args = ap.parse_args()

    if args.show_config:
        print(f"Файл настроек: {config.CONFIG_PATH}")
        print("существует" if config.CONFIG_PATH.exists() else "не создан — берутся значения по умолчанию")
        return 0
    if not args.transcript.exists():
        sys.exit(f"Файл не найден: {args.transcript}")

    try:
        _, text = analyze(args.transcript, args.words,
                          args.names.split(",") if args.names else None)
    except ValueError as e:
        sys.exit(str(e))

    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nСохранено: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
