#!/usr/bin/env python3
"""Распознавание через Nexara — проверка гипотезы про русскоязычные модели.

Whisper ломается системно на английских терминах в русской речи: пишет
«СПО» вместо «CFO» и «семью» вместо «CMO». ElevenLabs с этим справляется.
Открытый вопрос — что сделает система, обученная преимущественно на
русском: выиграет на самой речи или проиграет на терминах.

В единственном найденном замере на русских датасетах Nexara показала
лучший WER среди российских сервисов (0.391 против 0.550 у Yandex
SpeechKit), но тот тест шёл на тяжёлом аудио и не включал ни Whisper,
ни ElevenLabs — поэтому проверяем на своей записи.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

import transcript_utils as tu

API_URL = "https://api.nexara.ru/v1/audio/transcriptions"


def recognize(audio: Path, api_key: str, language: str, speakers: int | None,
              timeout: float) -> dict:
    data = {
        "task": "diarize",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    if language and language != "auto":
        data["language"] = language
    if speakers:
        data["num_speakers"] = str(speakers)

    size_mb = audio.stat().st_size / (1024 * 1024)
    print(f"🚀 Отправка в Nexara: {audio.name}")
    print(f"   Размер: {size_mb:.1f} МБ, спикеров: {speakers or 'авто'}")

    with audio.open("rb") as fh:
        resp = httpx.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": (audio.name, fh)},
            timeout=timeout,
        )

    if resp.status_code != 200:
        # Тело ответа печатаем, ключ — никогда.
        sys.exit(f"Nexara вернула {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def extract_words(payload: dict):
    """Слова с таймкодами и говорящим.

    Пословные метки приходят не всегда, поэтому при их отсутствии берём
    сегменты: для сверки важны текст и время, а не гранулярность.
    """
    words = []
    for w in payload.get("words") or []:
        if w.get("start") is None:
            continue
        words.append({
            "text": w.get("word") or w.get("text") or "",
            "start": w["start"], "end": w.get("end", w["start"]),
            "speaker": w.get("speaker"),
        })
    if words:
        return words, "пословно"

    for seg in payload.get("segments") or []:
        if seg.get("start") is None:
            continue
        words.append({
            "text": (seg.get("text") or "").strip(),
            "start": seg["start"], "end": seg.get("end", seg["start"]),
            "speaker": seg.get("speaker"),
        })
    return words, "по сегментам"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--reference", type=Path,
                    help="расшифровка WhisperX — под её разбивку подогнать текст")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--speakers", type=int)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--compare", action="store_true",
                    help="сразу свести с эталоном и пометить расхождения")
    args = ap.parse_args()

    if args.compare and not args.reference:
        ap.error("--compare требует --reference")
    api_key = os.environ.get("NEXARA_API_KEY")
    if not api_key:
        sys.exit(
            "Не задан NEXARA_API_KEY.\n"
            'Добавьте: echo \'export NEXARA_API_KEY="ключ"\' >> ~/.bash_profile\n'
            "и выполните: source ~/.bash_profile"
        )
    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    payload = recognize(args.audio, api_key, args.language, args.speakers, args.timeout)
    words, granularity = extract_words(payload)
    if not words:
        sys.exit(f"Nexara не вернула ни слов, ни сегментов. Ответ: {str(payload)[:400]}")

    speaker_of = tu.SpeakerMap()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = args.audio.with_name(f"{args.audio.stem}_nexara_{stamp}.txt")
    base.write_text(tu.native_transcript(words, speaker_of), encoding="utf-8")

    print(f"✅ Фрагментов: {len(words)} ({granularity}), голосов: {len(speaker_of)}")
    print(f"📄 Расшифровка Nexara: {base.name}")

    if args.reference:
        if not args.reference.exists():
            sys.exit(f"Эталон не найден: {args.reference}")
        reference = tu.parse_reference(args.reference)
        if not reference:
            sys.exit(f"В эталоне не найдено ни одной реплики: {args.reference.name}")
        fitted = base.with_name(f"{args.audio.stem}_nexara_свод_{stamp}.txt")
        fitted.write_text(tu.fitted_transcript(words, reference), encoding="utf-8")
        print(f"📐 Подогнано под разбивку эталона ({len(reference)} реплик): {fitted.name}")

        if args.compare:
            tu.run_comparison(args.reference, fitted, "nexara")
        else:
            print(f'\n  python compare_transcripts.py "{args.reference.name}" "{fitted.name}"')


if __name__ == "__main__":
    main()
