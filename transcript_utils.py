"""Общие части адаптеров к внешним системам распознавания.

Каждый провайдер отдаёт свой формат, но дальше с ним делается одно и то же:
привести к нашей разметке `[MM:SS - MM:SS] SPEAKER_XX:`, разложить по репликам
эталонной расшифровки и свести. Здесь живёт эта общая часть.
"""
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


def fitted_transcript(words, reference) -> str:
    """Раскладывает слова по репликам эталона.

    Сверять построчно можно только одинаково разбитые тексты, а каждая
    система режет речь по-своему. Раскладываем слова по временным окнам
    эталона — тогда сравнивается речь из одних и тех же отрезков записи.
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
