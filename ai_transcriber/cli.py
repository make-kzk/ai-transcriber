"""Единый консольный интерфейс (CLI) для ai-transcriber."""
import argparse
import sys
from pathlib import Path

from .core.compare import main as compare_main
from .core.pipeline import main as pipeline_main
from .engines import get_engine, AVAILABLE_ENGINES


def cli_transcribe(args):
    """Одиночный запуск одной системы распознавания."""
    engine = get_engine(args.engine)
    res = engine.transcribe(
        audio_path=args.audio,
        language=args.language,
        speakers=args.speakers,
        reference_path=args.reference,
        dry_run=args.dry_run,
    )
    if not res.success:
        sys.exit(f"Ошибка {res.label}: {res.error}")
    print(f"✅ Успешно выполнено: {res.label}")
    if res.transcript_path:
        print(f"📄 Текст: {res.transcript_path.name}")
    if res.words_path:
        print(f"🕐 Таймкоды: {res.words_path.name}")
    if res.fitted_path:
        print(f"📐 Свод: {res.fitted_path.name}")


def cli_analyze(args):
    """Анализ речи по готовой расшифровке."""
    from .speech_analysis import analyze
    names = [n.strip() for n in args.names.split(",")] if args.names else []
    try:
        data, text = analyze(
            transcript=args.transcript,
            audio=args.audio,
            words=args.words,
            names=names,
        )
        print(text)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            print(f"\nОтчет сохранен в: {args.output}")
    except Exception as e:
        sys.exit(f"Ошибка анализа речи: {e}")


def cli_gui(args):
    """Запуск графического интерфейса."""
    from .gui.app import main as gui_main
    gui_main()


def main():
    parser = argparse.ArgumentParser(
        prog="ai-transcriber",
        description="Мульти-движковая транскрибация, сверка и объективный анализ речи.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Команда для выполнения")

    # 1. pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Гибкий запуск нескольких систем со сверкой")
    p_pipe.add_argument("audio", type=Path, help="аудиофайл")
    p_pipe.add_argument("--systems", nargs="+", default=["whisper-v3", "elevenlabs"],
                        choices=list(AVAILABLE_ENGINES.keys()),
                        help="системы для запуска")
    p_pipe.add_argument("--base", choices=list(AVAILABLE_ENGINES.keys()),
                        help="базовая система (эталон) для нарезки реплик и сверки")
    p_pipe.add_argument("--stagger", type=float, default=2.0,
                        help="задержка (в секундах) между одновременными стартами (по умолчанию 2.0с)")
    p_pipe.add_argument("--language", default="ru", help="язык аудио")
    p_pipe.add_argument("--speakers", type=int, help="число спикеров")
    p_pipe.add_argument("--dry-run", action="store_true", help="план без запуска")
    p_pipe.add_argument("--no-compare", action="store_true", help="не выполнять сверку")
    p_pipe.add_argument("--output-dir", type=Path, help="директория сохранения")

    # 2. transcribe
    p_tx = subparsers.add_parser("transcribe", help="Быстрое распознавание одним движком")
    p_tx.add_argument("audio", type=Path, help="аудиофайл")
    p_tx.add_argument("--engine", default="whisper-v3", choices=list(AVAILABLE_ENGINES.keys()),
                      help="система распознавания (по умолчанию: whisper-v3)")
    p_tx.add_argument("--language", default="ru")
    p_tx.add_argument("--speakers", type=int)
    p_tx.add_argument("--reference", type=Path, help="эталон для подгонки")
    p_tx.add_argument("--dry-run", action="store_true")

    # 3. compare
    p_cmp = subparsers.add_parser("compare", help="Сверка двух готовых расшифровок")
    p_cmp.add_argument("base", type=Path, help="основа (эталон)")
    p_cmp.add_argument("other", type=Path, help="сверяемая расшифровка")
    p_cmp.add_argument("--noise", action="store_true", help="показывать шум")

    # 4. analyze
    p_an = subparsers.add_parser("analyze", help="Объективный анализ речи")
    p_an.add_argument("transcript", type=Path, help="файл расшифровки")
    p_an.add_argument("--audio", type=Path, help="аудиозапись (для Praat акустики)")
    p_an.add_argument("--words", type=Path, help="файл пословных таймкодов (*_слова.json)")
    p_an.add_argument("--names", type=str, help="имена спикеров через запятую ('Тая,Максим')")
    p_an.add_argument("--output", type=Path, help="сохранить отчет в файл")

    # 5. gui
    subparsers.add_parser("gui", help="Запустить графический интерфейс (Tkinter)")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "pipeline":
        # Передаем управление в pipeline
        from .core.pipeline import run_pipeline
        run_pipeline(
            audio=args.audio,
            systems=args.systems,
            base_system=args.base,
            stagger=args.stagger,
            language=args.language,
            speakers=args.speakers,
            dry_run=args.dry_run,
            compare=not args.no_compare,
            output_dir=args.output_dir,
        )
    elif args.subcommand == "transcribe":
        cli_transcribe(args)
    elif args.subcommand == "compare":
        from .core.compare import compare_files
        _, findings, out_file = compare_files(args.base, args.other, show_noise=args.noise)
        crit = sum(1 for f in findings if f["level"] == "critical")
        cont = sum(1 for f in findings if f["level"] == "content")
        print(f"Сверка сохранена: {out_file.name}")
        print(f"  критичных: {crit}, смысловых: {cont}")
    elif args.subcommand == "analyze":
        cli_analyze(args)
    elif args.subcommand == "gui":
        cli_gui(args)


if __name__ == "__main__":
    main()
