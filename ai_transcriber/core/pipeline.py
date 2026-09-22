"""Гибкий и кастомный оркестратор мульти-движкового пайплайна транскрибации."""
import argparse
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import utils
from .compare import compare_files

HERE = Path(__file__).resolve().parent.parent.parent

PIPELINE_ENGINES = {
    "whisper-v3": {"label": "Whisper large-v3", "kind": "whisper", "model": "large-v3"},
    "whisper-turbo": {"label": "Whisper large-v3-turbo", "kind": "whisper", "model": "large-v3-turbo"},
    "whisper-v2": {"label": "Whisper large-v2", "kind": "whisper", "model": "large-v2"},
    "elevenlabs": {"label": "ElevenLabs Scribe", "kind": "external"},
    "gemini": {"label": "Gemini 3.5 Transcribe", "kind": "external"},
}

_print_lock = threading.Lock()


def find_modal() -> str:
    """Поиск бинарника modal."""
    found = shutil.which("modal")
    if found:
        return found
    for c in (Path.home() / ".local/bin/modal", Path.home() / ".modal-venv/bin/modal"):
        if c.exists():
            return str(c)
    return "modal"


def step(text: str):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}", flush=True)


def fit_words_to_reference(words_file: Path, reference_file: Path, out_path: Path) -> Optional[Path]:
    """Раскладывает пословные таймкоды words_file по репликам reference_file."""
    if not words_file.exists() or not reference_file.exists():
        return None
    segments = utils.parse_reference(reference_file)
    if not segments:
        return None
    words = utils.load_words(words_file)
    fitted_text = utils.fitted_transcript(words, segments)
    out_path.write_text(fitted_text, encoding="utf-8")
    return out_path


def run_parallel_jobs(jobs: list[tuple[str, list[str]]], dry_run: bool, stagger: float = 2.0) -> dict[str, int]:
    """Запускает параллельные задачи с настраиваемой задержкой (stagger) между стартами."""
    results = {}

    def worker(label: str, cmd: list[str], delay: float):
        if delay > 0:
            time.sleep(delay)
        with _print_lock:
            print(f"  [{label}] 🚀 старт", flush=True)
        if dry_run:
            with _print_lock:
                print(f"  [{label}] (dry-run) " + " ".join(str(c) for c in cmd), flush=True)
            results[label] = 0
            return

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    with _print_lock:
                        print(f"  [{label}] {line}", flush=True)
        results[label] = proc.wait()

    threads = []
    for i, (label, cmd) in enumerate(jobs):
        t = threading.Thread(target=worker, args=(label, cmd, i * stagger))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


def run_pipeline(
    audio: Path,
    systems: list[str],
    base_system: Optional[str] = None,
    stagger: float = 2.0,
    language: str = "ru",
    speakers: Optional[int] = None,
    dry_run: bool = False,
    compare: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """Запускает кастомный параллельный пайплайн с настраиваемым выбором систем и задержкой."""
    audio = Path(audio).resolve()
    if not audio.exists() and not dry_run:
        raise FileNotFoundError(f"Файл не найден: {audio}")

    chosen = [k for k in systems if k in PIPELINE_ENGINES]
    if not chosen:
        raise ValueError(
            f"Не выбрано ни одной поддерживаемой системы. Запрошено: {systems}. "
            f"Доступны: {list(PIPELINE_ENGINES.keys())}"
        )

    out_dir = Path(output_dir).resolve() if output_dir else audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") if not dry_run else "<dry-run>"
    stem = audio.stem
    python_bin = sys.executable

    step(f"Запуск пайплайна: {len(chosen)} систем(ы), задержка {stagger:.1f}с между стартами")
    print(f"Файл: {audio.name}")
    print(f"Системы: {', '.join(PIPELINE_ENGINES[k]['label'] for k in chosen)}")

    jobs = []
    target_files = {}

    for key in chosen:
        cfg = PIPELINE_ENGINES[key]
        label = cfg["label"]

        if cfg["kind"] == "whisper":
            model = cfg["model"]
            out_txt = out_dir / f"{stem}_транскрибация_{model}_{stamp}.txt"
            target_files[key] = {
                "txt": out_txt,
                "words": out_dir / f"{stem}_транскрибация_{model}_{stamp}_слова.json",
            }
            cmd = [
                find_modal(), "run", str(HERE / "transcribe_modal.py"),
                "--file", str(audio),
                "--language", language,
                "--model", model,
                "--output", str(out_txt),
            ]
            if speakers:
                cmd += ["--speakers", str(speakers)]
            jobs.append((label, cmd))

        elif cfg["kind"] == "external":
            out_txt = out_dir / f"{stem}_{key}_{stamp}.txt"
            target_files[key] = {
                "txt": out_txt,
                "words": out_dir / f"{stem}_{key}_{stamp}_слова.json",
            }
            cmd = [python_bin, str(HERE / f"transcribe_{key}.py"), str(audio)]
            lang = {"gemini": {"ru": "ru-RU", "en": "en-US"}}.get(
                key, {"ru": "rus", "en": "eng"}
            ).get(language, language)
            cmd += ["--language", lang]
            if speakers and key != "gemini":
                cmd += ["--speakers", str(speakers)]
            jobs.append((label, cmd))

    # Запуск всех выбранных задач параллельно с задержкой stagger
    codes = run_parallel_jobs(jobs, dry_run=dry_run, stagger=stagger)

    produced = {}
    words_map = {}
    failed = []

    for key in chosen:
        label = PIPELINE_ENGINES[key]["label"]
        code = codes.get(label, 1)
        if code == 0:
            if dry_run:
                produced[key] = target_files[key]["txt"]
                words_map[key] = target_files[key]["words"]
                continue

            txt_path = target_files[key]["txt"]
            words_path = target_files[key]["words"]

            # Если файл с точным stamp не найден (например, адаптер пишет %Y-%m-%d_%H%M),
            # ищем самый свежий созданный файл по маске
            if not txt_path.exists():
                cfg = PIPELINE_ENGINES[key]
                if cfg["kind"] == "whisper":
                    pattern = f"{stem}_транскрибация_{cfg['model']}_*.txt"
                else:
                    pattern = f"{stem}_{key}_*.txt"
                matches = [
                    f for f in out_dir.glob(pattern)
                    if not f.name.endswith("_свод.txt") and "_свод_" not in f.name and not f.name.endswith("_слова.json")
                ]
                if matches:
                    txt_path = sorted(matches, key=lambda f: f.stat().st_mtime)[-1]

            if txt_path.exists():
                produced[key] = txt_path
                possible_words = txt_path.with_name(txt_path.stem + "_слова.json")
                if possible_words.exists():
                    words_map[key] = possible_words
                elif words_path.exists():
                    words_map[key] = words_path
                else:
                    words_map[key] = None
            else:
                failed.append(label)
        else:
            failed.append(label)

    # Определение базовой системы (референса)
    ref_key = None
    if base_system and base_system in produced:
        ref_key = base_system
    else:
        # По умолчанию: первая успешная модель Whisper, иначе первая успешная система
        for k in chosen:
            if k in produced and PIPELINE_ENGINES[k]["kind"] == "whisper":
                ref_key = k
                break
        if not ref_key and produced:
            ref_key = next(iter(produced))

    fitted = {}
    if ref_key and not dry_run:
        reference_txt = produced[ref_key]
        step(f"Приведение к эталонной сетке: {PIPELINE_ENGINES[ref_key]['label']}")
        for key, txt_path in produced.items():
            if key == ref_key:
                fitted[key] = txt_path
                continue
            words_file = words_map.get(key)
            if words_file and words_file.exists():
                fitted_path = out_dir / f"{stem}_{key}_свод_{stamp}.txt"
                res = fit_words_to_reference(words_file, reference_txt, fitted_path)
                if res:
                    fitted[key] = res
                    print(f"  ✅ {PIPELINE_ENGINES[key]['label']} приведен к репликам эталона: {fitted_path.name}")
                else:
                    print(f"  ⚠️ Не удалось подогнать {PIPELINE_ENGINES[key]['label']} под эталон.")
            else:
                print(f"  ⚠️ Нет пословных таймкодов для {PIPELINE_ENGINES[key]['label']} — сверка невозможна.")

    # Сверка между эталоном и вторичными системами
    comparisons = {}
    if compare and ref_key and len(fitted) >= 2 and not dry_run:
        step(f"Сверка расхождений с базой: {PIPELINE_ENGINES[ref_key]['label']}")
        base_path = fitted[ref_key]
        for key, other_path in fitted.items():
            if key == ref_key:
                continue
            try:
                _, findings, diff_file = compare_files(base_path, other_path)
                crit = sum(1 for f in findings if f["level"] == "critical")
                cont = sum(1 for f in findings if f["level"] == "content")
                comparisons[key] = {
                    "diff_file": diff_file,
                    "critical": crit,
                    "content": cont,
                }
                print(f"  🔍 {PIPELINE_ENGINES[key]['label']}: {crit} критичных, {cont} смысловых расхождений -> {diff_file.name}")
            except Exception as e:
                print(f"  ⚠️ Ошибка сверки с {PIPELINE_ENGINES[key]['label']}: {e}")

    step("Итоги работы пайплайна")
    for key in chosen:
        label = PIPELINE_ENGINES[key]["label"]
        if key in produced:
            status = "✅"
            info = produced[key].name if not dry_run else "(dry-run готов)"
            if key == ref_key:
                info += " [ЭТАЛОН]"
        else:
            status = "❌"
            info = "ошибка"
        print(f"  {status} {label:26} {info}")

    return {
        "chosen": chosen,
        "reference_system": ref_key,
        "produced": produced,
        "fitted": fitted,
        "comparisons": comparisons,
        "failed": failed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="путь к аудиофайлу")
    ap.add_argument("--systems", nargs="+", default=["whisper-v3", "elevenlabs"],
                    choices=list(PIPELINE_ENGINES.keys()),
                    help="какие системы запустить (по умолчанию: whisper-v3 elevenlabs)")
    ap.add_argument("--base", choices=list(PIPELINE_ENGINES.keys()),
                    help="какую систему назначить эталоном для реплик и сверки")
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="задержка в секундах между стартами параллельных задач (по умолчанию 2.0с)")
    ap.add_argument("--language", default="ru", help="язык аудио (ru, en)")
    ap.add_argument("--speakers", type=int, help="количество говорящих")
    ap.add_argument("--dry-run", action="store_true",
                    help="показать план без запуска и без расходов")
    ap.add_argument("--no-compare", action="store_true", help="не выполнять сверку расхождений")
    ap.add_argument("--output-dir", type=Path, help="папка для сохранения результатов")
    args = ap.parse_args()

    try:
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
    except Exception as e:
        sys.exit(f"Ошибка пайплайна: {e}")


if __name__ == "__main__":
    main()
