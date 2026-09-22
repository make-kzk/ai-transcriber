#!/usr/bin/env python3
"""Распознавание через Gemini 3.5 Transcribe.

Отдельная модель распознавания, а не общая: диаризация, пословные таймкоды
и — главное для нашего случая — подсказка словаря. Whisper, Nexara и
Deepgram одинаково ломались на английских терминах в русской речи («СПО»
вместо «CFO», «Семео» вместо «CMO»). Здесь можно назвать эти термины
заранее, до распознавания.

Список по умолчанию собран из тех мест, где системы ошибались на реальной
записи, и дополняется через --vocabulary.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import transcript_utils as tu

MODEL = "gemini-3.5-transcribe"

# Термины, на которых спотыкались все проверенные системы.
DEFAULT_VOCABULARY = [
    "CEO", "CFO", "CMO", "CPO", "CTO", "COO", "C-level",
    "HR", "HRD", "hiring manager", "job offer", "offer",
    "iGaming", "fintech", "blockchain", "e-com", "DevOps",
    "backend", "frontend", "R&D", "LinkedIn", "onboarding",
    "welcome-встреча", "релокация", "испытательный срок",
]


def _seconds(value) -> float | None:
    """Смещения приходят по-разному: числом или строкой вида «1.5s»."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("s")
    try:
        return float(text)
    except ValueError:
        return None


def recognize(audio: Path, language: str, vocabulary: list, model: str,
              on_progress=None):
    from google import genai
    from google.genai import types

    def say(text):
        print(text, flush=True)
        if on_progress:
            on_progress(text)

    client = genai.Client()
    say(f"🚀 Загрузка в Gemini: {audio.name} "
        f"({audio.stat().st_size / (1024*1024):.1f} МБ)")
    uploaded = client.files.upload(file=str(audio))

    # Большая запись обрабатывается сервисом не мгновенно.
    deadline = time.time() + 900
    while uploaded.state == types.FileState.PROCESSING and time.time() < deadline:
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state != types.FileState.ACTIVE:
        sys.exit(f"Запись не принята сервисом: состояние {uploaded.state}")

    say(f"   Распознаю, словарь-подсказка: {len(vocabulary)} терминов")
    interaction = client.interactions.create(
        model=model,
        input=[{"type": "audio", "uri": uploaded.uri,
                "mime_type": uploaded.mime_type}],
        generation_config={
            "transcription_config": {
                "language_codes": [language],
                "custom_vocabulary": vocabulary,
                # Пословные метки и диаризация требуют verbatim:
                # в режиме smart они недоступны.
                "mode": {
                    "type": "verbatim",
                    "diarization_mode": "speaker",
                    "timestamp_granularities": ["word"],
                },
            }
        },
    )
    return interaction


def extract_words(interaction):
    """Слова с таймкодами и говорящим из шагов ответа."""
    words = []
    for step in getattr(interaction, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                if getattr(ann, "type", None) != "word_info":
                    continue
                start = _seconds(getattr(ann, "start_offset", None))
                end = _seconds(getattr(ann, "end_offset", None))
                if start is None:
                    continue
                words.append({"text": getattr(ann, "text", ""),
                              "start": start, "end": end if end is not None else start,
                              "speaker": getattr(ann, "speaker", None)})
    return words


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--reference", type=Path,
                    help="расшифровка WhisperX — под её разбивку подогнать текст")
    ap.add_argument("--language", default="ru-RU")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--vocabulary", type=Path,
                    help="файл со словарём-подсказкой, по слову или обороту в строке")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare and not args.reference:
        ap.error("--compare требует --reference")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Не задан GEMINI_API_KEY — ключ выдаётся в Google AI Studio.")
    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    vocabulary = list(DEFAULT_VOCABULARY)
    if args.vocabulary and args.vocabulary.exists():
        vocabulary += [l.strip() for l in
                       args.vocabulary.read_text(encoding="utf-8").splitlines() if l.strip()]
    vocabulary = list(dict.fromkeys(vocabulary))[:1000]   # предел сервиса

    interaction = recognize(args.audio, args.language, vocabulary, args.model)
    words = extract_words(interaction)
    if not words:
        sys.exit("Gemini не вернул пословной разметки — проверьте, что режим verbatim принят.")

    speaker_of = tu.SpeakerMap()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = args.audio.with_name(f"{args.audio.stem}_gemini_{stamp}.txt")
    base.write_text(tu.native_transcript(words, speaker_of), encoding="utf-8")
    print(f"✅ Слов: {len(words)}, голосов: {len(speaker_of)}")
    print(f"📄 Расшифровка Gemini: {base.name}")
    print(f"🕐 Пословные таймкоды: {tu.save_words(words, base).name}")

    if args.reference:
        if not args.reference.exists():
            sys.exit(f"Эталон не найден: {args.reference}")
        reference = tu.parse_reference(args.reference)
        if not reference:
            sys.exit("В эталоне нет ни одной реплики.")
        fitted = base.with_name(f"{args.audio.stem}_gemini_свод_{stamp}.txt")
        fitted.write_text(tu.fitted_transcript(words, reference), encoding="utf-8")
        print(f"📐 Подогнано под эталон ({len(reference)} реплик): {fitted.name}")
        if args.compare:
            tu.run_comparison(args.reference, fitted, "gemini")


if __name__ == "__main__":
    main()
