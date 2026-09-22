"""Общие части адаптеров к внешним системам распознавания.

Каждый провайдер отдаёт свой формат, но дальше с ним делается одно и то же:
привести к нашей разметке `[MM:SS - MM:SS] SPEAKER_XX:`, разложить по репликам
эталонной расшифровки и свести. Здесь живёт эта общая часть.
"""
import json
import re
import sys
from pathlib import Path

SEG_RE = re.compile(r"^\[(\d\d):(\d\d) - (\d\d):(\d\d)\] (SPEAKER_\d+):$")


def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class SpeakerMap:
    """Метки говорящих у всех разные: speaker_0, speaker_1, «Агент».

    Нумеруем по порядку появления — так не зависим от того, с нуля или с
    единицы считает провайдер, и переживаем строковые роли.
    """

    def __init__(self):
        self._seen = {}

    def __call__(self, raw) -> str:
        key = str(raw) if raw is not None else "—"
        if key not in self._seen:
            self._seen[key] = f"SPEAKER_{len(self._seen):02d}"
        return self._seen[key]

    def __len__(self):
        return len(self._seen)


def native_transcript(words, speaker_of) -> str:
    """Расшифровка в нашем формате: подряд идущие слова одного голоса — одна реплика."""
    out, speaker, buf, start, end = [], None, [], 0.0, 0.0

    def flush():
        if buf:
            out.append(f"[{mmss(start)} - {mmss(end)}] {speaker}:\n{' '.join(buf).strip()}\n")

    for w in words:
        spk = speaker_of(w.get("speaker"))
        if spk != speaker:
            flush()
            speaker, buf, start = spk, [], w["start"]
        buf.append(w["text"])
        end = w["end"]
    flush()
    return "\n".join(out)


def parse_reference(path: Path):
    """Реплики эталонной расшифровки: границы по времени и заголовок."""
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


def save_words(words, transcript_path: Path) -> Path:
    """Пословные таймкоды рядом с расшифровкой.

    Нужны для пауз и запинок внутри реплики: в самой расшифровке время
    указано только на границах реплик, а внутри неё всё склеено.
    """
    out = transcript_path.with_name(transcript_path.stem + "_слова.json")
    out.write_text(
        json.dumps([{k: w.get(k) for k in ("text", "start", "end", "speaker")}
                    for w in words], ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def parse_transcript(path: Path):
    """Реплики с временем, говорящим и текстом — для разбора, а не для сверки."""
    segs, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
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
    for seg in segs:
        seg["text"] = " ".join(seg["lines"])
        del seg["lines"]
    return segs


def fitted_transcript(words, reference) -> str:
    """Раскладывает слова по репликам эталона.

    Сверять построчно можно только одинаково разбитые тексты, а каждая
    система режет речь по-своему. Раскладываем слова по временным окнам
    эталона — тогда сравнивается речь из одних и тех же отрезков записи.
    """
    buckets, orphans = [[] for _ in reference], 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        for i, seg in enumerate(reference):
            if seg["start"] <= mid <= seg["end"] + 1:
                buckets[i].append(w["text"])
                break
        else:
            # Слово прозвучало там, где у эталона реплики нет. Сверка его не
            # увидит: так теряются места, где вторая система услышала то,
            # чего эталон не заметил вовсе.
            orphans += 1

    if orphans:
        share = orphans / len(words) * 100
        note = "" if share < 1 else "  ⚠️ эталон мог пропустить эти фрагменты"
        print(f"   вне реплик эталона осталось слов: {orphans} ({share:.1f}%){note}")

    out = []
    for seg, bucket in zip(reference, buckets):
        out += [seg["head"], " ".join(bucket).strip() or "—", ""]
    return "\n".join(out)


def run_comparison(base: Path, other: Path, suffix: str):
    """Сводит расшифровки и помечает расхождения тем же кодом, что и вручную."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import compare_transcripts as ct
    except ImportError:
        print("Не найден compare_transcripts.py — сверка пропущена.")
        return

    base_segs, other_segs = ct.parse(base), ct.parse(other)
    body, findings = ct.compare(base_segs, other_segs)
    header = ct.render_header(findings, base.name, other.name, base_segs)
    out = base.with_name(f"{base.stem}_сверка_{suffix}.txt")
    out.write_text(header + body, encoding="utf-8")

    crit = sum(1 for f in findings if f["level"] == "critical")
    cont = sum(1 for f in findings if f["level"] == "content")
    print("\n🔍 Сверка с независимой системой:")
    print(f"   критичных расхождений: {crit}")
    print(f"   смысловых разночтений: {cont}")
    print(f"📋 {out.name}")
