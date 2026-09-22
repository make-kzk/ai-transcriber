#!/usr/bin/env python3
"""Прогоняет запись через выбранные системы распознавания и сводит результаты.

Порядок в SYSTEMS — это и порядок предпочтения: первая отмеченная система
становится основой сводного документа, остальные идут в сверку. ElevenLabs
стоит первым не случайно: на проверке четырёх систем он единственный верно
взял аббревиатуры (CFO, CMO) и не потерял отрицание в ключевой фразе.

Разбивку на реплики и диаризацию даёт Whisper — внешние системы режут речь
по-своему, и чтобы тексты можно было приложить друг к другу построчно, их
слова раскладываются по репликам первой выбранной модели Whisper.
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Порядок важен: он задаёт, кто станет основой документа.
SYSTEMS = {
    "gemini": {"label": "Gemini 3.5 Transcribe", "kind": "external"},
    "elevenlabs": {"label": "ElevenLabs Scribe", "kind": "external"},
    "whisper-v3": {"label": "Whisper large-v3", "kind": "whisper", "model": "large-v3"},
    "whisper-turbo": {"label": "Whisper large-v3-turbo", "kind": "whisper",
                      "model": "large-v3-turbo"},
    "whisper-v2": {"label": "Whisper large-v2", "kind": "whisper", "model": "large-v2"},
}


def find_modal() -> str:
    found = shutil.which("modal")
    if found:
        return found
    for c in (Path.home() / ".local/bin/modal", Path.home() / ".modal-venv/bin/modal"):
        if c.exists():
            return str(c)
    sys.exit("Не найдена утилита modal. Запустите install.sh")


def step(text: str):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}", flush=True)


def run(cmd, dry: bool) -> int:
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--systems", nargs="+", required=True, choices=list(SYSTEMS),
                    help="какие системы запустить")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--speakers", type=int)
    ap.add_argument("--dry-run", action="store_true",
                    help="показать план без запуска и без расходов")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    # Порядок галочек не важен — восстанавливаем порядок предпочтения.
    chosen = [k for k in SYSTEMS if k in args.systems]
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = args.audio.stem
    python = sys.executable

    whisper_keys = [k for k in chosen if SYSTEMS[k]["kind"] == "whisper"]
    reference = None
    produced = {}   # система -> файл, готовый к построчной сверке
    failed = []

    # 1. Модели Whisper: каждая своим прогоном в Modal.
    for key in whisper_keys:
        model = SYSTEMS[key]["model"]
        out = args.audio.with_name(f"{stem}_транскрибация_{model}_{stamp}.txt")
        step(f"{SYSTEMS[key]['label']}")
        cmd = [find_modal(), "run", str(HERE / "transcribe_modal.py"),
               "--file", str(args.audio), "--language", args.language,
               "--model", model, "--output", str(out)]
        if args.speakers:
            cmd += ["--speakers", str(args.speakers)]
        if run(cmd, args.dry_run) != 0:
            failed.append(SYSTEMS[key]["label"])
            continue
        produced[key] = out
        if reference is None:
            # Первая удачная модель Whisper задаёт разбивку для всех остальных.
            reference = out

    # 2. Внешние системы: подгоняются под разбивку Whisper, если она есть.
    for key in chosen:
        if SYSTEMS[key]["kind"] != "external":
            continue
        step(SYSTEMS[key]["label"])
        cmd = [python, str(HERE / f"transcribe_{key}.py"), str(args.audio)]
        # Коды языка у сервисов разные: ISO-639-3 у ElevenLabs, BCP-47 у Gemini.
        lang = {"gemini": {"ru": "ru-RU", "en": "en-US"},
                }.get(key, {"ru": "rus", "en": "eng"}).get(args.language, args.language)
        cmd += ["--language", lang]
        if args.speakers and key != "gemini":
            cmd += ["--speakers", str(args.speakers)]
        if reference:
            cmd += ["--reference", str(reference)]
        else:
            print("  Модели Whisper не выбраны — сверять будет не с чем.", flush=True)
        if run(cmd, args.dry_run) != 0:
            failed.append(SYSTEMS[key]["label"])
            continue
        if reference and not args.dry_run:
            fits = sorted(args.audio.parent.glob(f"{stem}_{key}_свод_*.txt"),
                          key=lambda f: f.stat().st_mtime)
            if fits:
                produced[key] = fits[-1]
        elif args.dry_run:
            produced[key] = args.audio.with_name(f"{stem}_{key}_свод_<время>.txt")

    # 3. Сверка: основа — первая по порядку предпочтения из удавшихся.
    ready = [k for k in chosen if k in produced]
    if len(ready) >= 2:
        base_key, base = ready[0], produced[ready[0]]
        step(f"Сверка — основа: {SYSTEMS[base_key]['label']}")
        for key in ready[1:]:
            cmd = [python, str(HERE / "compare_transcripts.py"),
                   str(base), str(produced[key])]
            if run(cmd, args.dry_run) != 0:
                print(f"  Не удалось сверить с {SYSTEMS[key]['label']}.", flush=True)
    elif len(ready) == 1:
        print("\nВыбрана одна система — сверять не с чем.", flush=True)

    step("Итог")
    for key in chosen:
        mark = "✅" if key in produced else "❌"
        name = produced[key].name if key in produced else "не получилось"
        print(f"  {mark} {SYSTEMS[key]['label']:24} {name}")
    if failed:
        print(f"\n⚠️  Не отработали: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
