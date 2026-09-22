#!/usr/bin/env python3
"""Распознавание через Deepgram Nova-3 — последний кандидат на сверку.

Главное здесь — режим `multi`. В отличие от остальных систем, Nova-3 умеет
распознавать несколько языков одновременно, и в этот набор входят русский и
английский. Ваш материал — русская речь с английскими терминами, и именно
на них ломались Whisper и Nexara: «СПО» вместо «CFO», «Семео» вместо «CMO».
Режим multi существует ровно для такой смешанной речи, поэтому берём его
по умолчанию; `--language ru` оставлен для сравнения.

Диаризация для записей включена бесплатно, файл уходит одним запросом
без многочастной формы — Deepgram принимает сырые байты.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

import transcript_utils as tu

API_URL = "https://api.deepgram.com/v1/listen"

# Явная таблица вместо mimetypes: для .m4a он выдаёт audio/mp4a-latm,
# которого Deepgram не ждёт.
CONTENT_TYPES = {
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/aac",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
}


def recognize(audio: Path, api_key: str, language: str, model: str,
              timeout: float) -> dict:
    params = {
        "model": model,
        "language": language,
        "diarize_model": "latest",   # включает диаризацию и выбирает версию модели
        "punctuate": "true",
        "smart_format": "true",
        "utterances": "true",
    }
    content_type = CONTENT_TYPES.get(audio.suffix.lower(), "audio/*")

    size_mb = audio.stat().st_size / (1024 * 1024)
    print(f"🚀 Отправка в Deepgram: {audio.name}")
    print(f"   Размер: {size_mb:.1f} МБ, модель: {model}, язык: {language}")

    resp = httpx.post(
        API_URL,
        params=params,
        headers={"Authorization": f"Token {api_key}", "Content-Type": content_type},
        content=audio.read_bytes(),
        timeout=timeout,
    )
    if resp.status_code != 200:
        # Тело ответа печатаем, ключ — никогда.
        sys.exit(f"Deepgram вернул {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def extract_words(payload: dict):
    """Слова с таймкодами и номером говорящего."""
    try:
        alt = payload["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError):
        return []
    return [
        {
            # punctuated_word приходит при smart_format и читается лучше сырого.
            "text": w.get("punctuated_word") or w.get("word", ""),
            "start": w["start"],
            "end": w["end"],
            "speaker": w.get("speaker"),
        }
        for w in alt.get("words", [])
        if w.get("start") is not None
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--reference", type=Path,
                    help="расшифровка WhisperX — под её разбивку подогнать текст")
    ap.add_argument("--language", default="multi",
                    help="multi — смешанная речь (по умолчанию), ru — только русский")
    ap.add_argument("--model", default="nova-3")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--compare", action="store_true",
                    help="сразу свести с эталоном и пометить расхождения")
    args = ap.parse_args()

    if args.compare and not args.reference:
        ap.error("--compare требует --reference")
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        sys.exit(
            "Не задан DEEPGRAM_API_KEY.\n"
            'Добавьте: echo \'export DEEPGRAM_API_KEY="ключ"\' >> ~/.bash_profile\n'
            "и выполните: source ~/.bash_profile"
        )
    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    payload = recognize(args.audio, api_key, args.language, args.model, args.timeout)
    words = extract_words(payload)
    if not words:
        sys.exit(f"Deepgram не вернул слов. Ответ: {str(payload)[:400]}")

    speaker_of = tu.SpeakerMap()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = f"deepgram-{args.language}"
    base = args.audio.with_name(f"{args.audio.stem}_{tag}_{stamp}.txt")
    base.write_text(tu.native_transcript(words, speaker_of), encoding="utf-8")

    duration = payload.get("metadata", {}).get("duration")
    print(f"✅ Распознано слов: {len(words)}, голосов: {len(speaker_of)}"
          + (f", длительность: {tu.mmss(duration)}" if duration else ""))
    print(f"📄 Расшифровка Deepgram: {base.name}")
    print(f"🕐 Пословные таймкоды: {tu.save_words(words, base).name}")

    if args.reference:
        if not args.reference.exists():
            sys.exit(f"Эталон не найден: {args.reference}")
        reference = tu.parse_reference(args.reference)
        if not reference:
            sys.exit(f"В эталоне не найдено ни одной реплики: {args.reference.name}")
        fitted = base.with_name(f"{args.audio.stem}_{tag}_свод_{stamp}.txt")
        fitted.write_text(tu.fitted_transcript(words, reference), encoding="utf-8")
        print(f"📐 Подогнано под разбивку эталона ({len(reference)} реплик): {fitted.name}")

        if args.compare:
            tu.run_comparison(args.reference, fitted, tag)
        else:
            print(f'\n  python compare_transcripts.py "{args.reference.name}" "{fitted.name}"')


if __name__ == "__main__":
    main()
