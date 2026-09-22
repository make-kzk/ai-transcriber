"""Утилиты разметки, нормализации и приведения расшифровок к репликам."""
import json
import re
from pathlib import Path
from typing import Optional

from .models import Word, Segment

SEG_RE = re.compile(r"^\[(\d\d):(\d\d) - (\d\d):(\d\d)\] (SPEAKER_\d+):$")


def mmss(seconds: float) -> str:
    """Форматирует секунды в MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def clean_line(text: str) -> str:
    """Очистка строки от лишних пробелов."""
    return re.sub(r"\s+", " ", text).strip()


class SpeakerMap:
    """Нормализация меток говорящих к стандартным SPEAKER_00, SPEAKER_01 и т.д."""

    def __init__(self):
        self._seen: dict[str, str] = {}

    def __call__(self, raw: Optional[str]) -> str:
        key = str(raw) if raw is not None else "—"
        if key not in self._seen:
            self._seen[key] = f"SPEAKER_{len(self._seen):02d}"
        return self._seen[key]

    def __len__(self) -> int:
        return len(self._seen)


def native_transcript(words: list[dict], speaker_of: SpeakerMap) -> str:
    """Собирает расшифровку: подряд идущие слова одного спикера объединяются в реплику."""
    out, speaker, buf, start, end = [], None, [], 0.0, 0.0

    def flush():
        if buf:
            out.append(f"[{mmss(start)} - {mmss(end)}] {speaker}:\n{' '.join(buf).strip()}\n")

    for w in words:
        spk = speaker_of(w.get("speaker"))
        if spk != speaker:
            flush()
            speaker, buf, start = spk, [], float(w["start"])
        buf.append(str(w["text"]))
        end = float(w["end"])
    flush()
    return "\n".join(out)


format_transcript = native_transcript


def parse_reference(path: Path) -> list[dict]:
    """Извлекает границы реплик эталонной расшифровки."""
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            segs.append({
                "head": line.strip(),
                "start": int(m.group(1)) * 60 + int(m.group(2)),
                "end": int(m.group(3)) * 60 + int(m.group(4)),
                "speaker": m.group(5),
            })
    return segs


def parse_transcript(path: Path) -> list[Segment]:
    """Парсит расшифровку в типизированные объекты Segment."""
    segs, cur = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            cur = {
                "start": int(m.group(1)) * 60 + int(m.group(2)),
                "end": int(m.group(3)) * 60 + int(m.group(4)),
                "speaker": m.group(5),
                "lines": [],
            }
            segs.append(cur)
        elif line.strip() and cur is not None:
            cur["lines"].append(line.strip())

    result = []
    for s in segs:
        result.append(
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                speaker=s["speaker"],
                text=" ".join(s["lines"]),
            )
        )
    return result


def save_words(words: list[dict], transcript_path: Path) -> Path:
    """Сохраняет пословные таймкоды рядом с расшифровкой."""
    out = transcript_path.with_name(transcript_path.stem + "_слова.json")
    out.write_text(
        json.dumps(
            [{k: w.get(k) for k in ("text", "start", "end", "speaker")} for w in words],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def load_words(path: Path) -> list[dict]:
    """Читает пословные таймкоды из JSON-файла."""
    out = []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for w in raw:
        if w.get("start") is None:
            continue
        out.append({
            "text": w.get("text") or w.get("word") or "",
            "start": float(w["start"]),
            "end": float(w.get("end") or w["start"]),
            "speaker": w.get("speaker"),
        })
    return out


def assign_words_to_segments(words: list[dict], reference: list[dict]) -> tuple[list[list[str]], int]:
    """Распределяет слова по временным сегментам эталона."""
    buckets: list[list[str]] = [[] for _ in reference]
    orphans = 0
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        for i, seg in enumerate(reference):
            if seg["start"] <= mid <= seg["end"] + 1.0:
                buckets[i].append(str(w["text"]))
                break
        else:
            orphans += 1
    return buckets, orphans


def fitted_transcript(words: list[dict], reference: list[dict]) -> str:
    """Формирует сводный текст, разложенный по репликам эталона."""
    buckets, orphans = assign_words_to_segments(words, reference)
    if orphans and len(words) > 0:
        share = orphans / len(words) * 100.0
        note = "" if share < 1.0 else "  ⚠️ эталон мог пропустить эти фрагменты"
        print(f"   вне реплик эталона осталось слов: {orphans} ({share:.1f}%){note}")

    out = []
    for seg, bucket in zip(reference, buckets):
        out += [seg["head"], " ".join(bucket).strip() or "—", ""]
    return "\n".join(out)
