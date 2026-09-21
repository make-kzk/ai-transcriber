#!/usr/bin/env python3
"""Распознавание через ElevenLabs Scribe — для независимой сверки с WhisperX.

Смысл не в замене основного пайплайна, а в том, что это другая система.
Две версии Whisper ошибаются в одних и тех же местах: large-v2 и large-v3
одинаково переврали «многогранный опыт» и «роадмап», и сверка их между собой
такие места не видит. ElevenLabs обучен отдельно, его ошибки не совпадают
с ошибками Whisper — значит несогласие заметно там, где раньше было слепое
пятно.

Пишет два файла:
  *_elevenlabs_<время>.txt   — своя разбивка и своя диаризация,
                               чтобы оценить ElevenLabs сам по себе;
  *_elevenlabs_свод.txt      — тот же текст, разложенный по репликам
                               эталонной расшифровки, для compare_transcripts.py.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

import transcript_utils as tu

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def recognize(audio: Path, api_key: str, language: str, speakers: int | None,
              model: str, timeout: float):
    """Отправляет файл в Scribe и возвращает разобранный ответ."""
    data = {"model_id": model, "diarize": "true", "timestamps_granularity": "word"}
    if language and language != "auto":
        data["language_code"] = language
    if speakers:
        data["num_speakers"] = str(speakers)

    size_mb = audio.stat().st_size / (1024 * 1024)
    print(f"🚀 Отправка в ElevenLabs Scribe: {audio.name}")
    print(f"   Размер: {size_mb:.1f} МБ, модель: {model}, спикеров: {speakers or 'авто'}")

    with audio.open("rb") as fh:
        resp = httpx.post(
            API_URL,
            headers={"xi-api-key": api_key},
            data=data,
            files={"file": (audio.name, fh)},
            timeout=timeout,
        )

    if resp.status_code != 200:
        # Тело ответа печатаем, ключ — никогда.
        sys.exit(f"ElevenLabs вернул {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def words_of(payload):
    """Только слова: разметка пауз и звуковых событий нам не нужна."""
    return [
        {"text": w["text"], "start": w["start"], "end": w["end"],
         "speaker": w.get("speaker_id")}
        for w in payload.get("words", [])
        if w.get("type") == "word" and w.get("start") is not None
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="аудиофайл")
    ap.add_argument("--reference", type=Path,
                    help="расшифровка WhisperX — под её разбивку подогнать текст")
    ap.add_argument("--language", default="rus", help="код языка ISO-639-3 (по умолчанию rus)")
    ap.add_argument("--speakers", type=int, help="сколько голосов в записи")
    ap.add_argument("--model", default="scribe_v2", help="модель ElevenLabs")
    ap.add_argument("--timeout", type=float, default=900.0, help="таймаут запроса, секунд")
    ap.add_argument("--compare", action="store_true",
                    help="сразу свести с эталоном и пометить расхождения")
    args = ap.parse_args()

    if args.compare and not args.reference:
        ap.error("--compare требует --reference")

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit(
            "Не задан ELEVENLABS_API_KEY.\n"
            'Добавьте: echo \'export ELEVENLABS_API_KEY="ключ"\' >> ~/.zshrc\n'
            "и откройте новый терминал."
        )
    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    payload = recognize(args.audio, api_key, args.language, args.speakers,
                        args.model, args.timeout)
    words = words_of(payload)
    duration = payload.get("audio_duration_secs")
    speaker_of = tu.SpeakerMap()
    print(f"✅ Распознано слов: {len(words)}"
          + (f", длительность: {tu.mmss(duration)}" if duration else ""))

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = args.audio.with_name(f"{args.audio.stem}_elevenlabs_{stamp}.txt")
    base.write_text(tu.native_transcript(words, speaker_of), encoding="utf-8")
    print(f"📄 Расшифровка ElevenLabs: {base.name}")

    if args.reference:
        if not args.reference.exists():
            sys.exit(f"Эталон не найден: {args.reference}")
        reference = tu.parse_reference(args.reference)
        if not reference:
            sys.exit(f"В эталоне не найдено ни одной реплики: {args.reference.name}")
        fitted = base.with_name(f"{args.audio.stem}_elevenlabs_свод_{stamp}.txt")
        fitted.write_text(tu.fitted_transcript(words, reference), encoding="utf-8")
        print(f"📐 Подогнано под разбивку эталона ({len(reference)} реплик): {fitted.name}")

        if args.compare:
            tu.run_comparison(args.reference, fitted, "elevenlabs")
        else:
            print("\nТеперь сверка:")
            print(f'  python compare_transcripts.py "{args.reference.name}" "{fitted.name}"')


if __name__ == "__main__":
    main()
