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
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SEG_RE = re.compile(r"^\[(\d\d):(\d\d) - (\d\d):(\d\d)\] (SPEAKER_\d+):$")


def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


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
        w for w in payload.get("words", [])
        if w.get("type") == "word" and w.get("start") is not None
    ]


def native_transcript(words):
    """Разбивка и диаризация самого ElevenLabs."""
    out, speaker, buf, start, end = [], None, [], 0.0, 0.0

    def flush():
        if buf:
            out.append(f"[{mmss(start)} - {mmss(end)}] {speaker}:\n{' '.join(buf).strip()}\n")

    for w in words:
        spk = normalize_speaker(w.get("speaker_id"))
        if spk != speaker:
            flush()
            speaker, buf, start = spk, [], w["start"]
        buf.append(w["text"])
        end = w["end"]
    flush()
    return "\n".join(out)


def normalize_speaker(raw) -> str:
    """speaker_1 -> SPEAKER_00, чтобы формат совпадал с нашим."""
    if not raw:
        return "SPEAKER_00"
    m = re.search(r"(\d+)", str(raw))
    n = int(m.group(1)) if m else 0
    return f"SPEAKER_{max(0, n - 1):02d}"


def parse_reference(path: Path):
    """Реплики эталонной расшифровки: границы по времени и метка спикера."""
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            segs.append({
                "head": line.strip(),
                "start": int(m.group(1)) * 60 + int(m.group(2)),
                "end": int(m.group(3)) * 60 + int(m.group(4)),
            })
    return segs


def fitted_transcript(words, reference):
    """Раскладывает слова ElevenLabs по репликам эталона.

    Сверять построчно можно только расшифровки с одинаковой разбивкой,
    а ElevenLabs режет речь по-своему. Берём его слова и распределяем
    по временным окнам эталона — тогда сравнивается текст в одних и тех
    же отрезках записи, а не случайно совпавшие куски.
    """
    buckets = [[] for _ in reference]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        for i, seg in enumerate(reference):
            if seg["start"] <= mid <= seg["end"] + 1:
                buckets[i].append(w["text"])
                break

    out = []
    for seg, bucket in zip(reference, buckets):
        out.append(seg["head"])
        out.append(" ".join(bucket).strip() or "—")
        out.append("")
    return "\n".join(out)


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
    speakers_found = len({w.get("speaker_id") for w in words})
    print(f"✅ Распознано слов: {len(words)}, голосов: {speakers_found}"
          + (f", длительность: {mmss(duration)}" if duration else ""))

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = args.audio.with_name(f"{args.audio.stem}_elevenlabs_{stamp}.txt")
    base.write_text(native_transcript(words), encoding="utf-8")
    print(f"📄 Расшифровка ElevenLabs: {base.name}")

    if args.reference:
        if not args.reference.exists():
            sys.exit(f"Эталон не найден: {args.reference}")
        reference = parse_reference(args.reference)
        if not reference:
            sys.exit(f"В эталоне не найдено ни одной реплики: {args.reference.name}")
        fitted = base.with_name(f"{args.audio.stem}_elevenlabs_свод_{stamp}.txt")
        fitted.write_text(fitted_transcript(words, reference), encoding="utf-8")
        print(f"📐 Подогнано под разбивку эталона ({len(reference)} реплик): {fitted.name}")

        if args.compare:
            run_comparison(args.reference, fitted)
        else:
            print("\nТеперь сверка:")
            print(f'  python compare_transcripts.py "{args.reference.name}" "{fitted.name}"')


def run_comparison(base: Path, other: Path):
    """Сводит расшифровки и помечает расхождения — тем же кодом, что и вручную."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import compare_transcripts as ct
    except ImportError:
        print("Не найден compare_transcripts.py — сверка пропущена.")
        return

    base_segs, other_segs = ct.parse(base), ct.parse(other)
    body, findings = ct.compare(base_segs, other_segs)
    header = ct.render_header(findings, base.name, other.name, base_segs)
    out = base.with_name(base.stem + "_сверка_elevenlabs.txt")
    out.write_text(header + body, encoding="utf-8")

    crit = sum(1 for f in findings if f["level"] == "critical")
    cont = sum(1 for f in findings if f["level"] == "content")
    print(f"\n🔍 Сверка с независимой системой:")
    print(f"   критичных расхождений: {crit}")
    print(f"   смысловых разночтений: {cont}")
    print(f"📋 {out.name}")


if __name__ == "__main__":
    main()
