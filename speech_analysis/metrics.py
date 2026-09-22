"""Вычисление метрик. Ничего не печатает и ни о чём не догадывается."""
import json
import re
import statistics as st
from pathlib import Path

# Формат расшифровки — единственное, что связывает модуль с остальной
# системой. Это стабильный контракт, поэтому разбор свой: пакет не должен
# зависеть от кода транскрибации, чтобы дорабатываться отдельно.
SEG_RE = re.compile(r"^\[(\d\d):(\d\d) - (\d\d):(\d\d)\] (SPEAKER_\d+):$")

# Не является нормальным словом: тянущийся гласный, «м», слог через дефис.
# Простой набор букв ловил бы «а», «у», «на», «ну».
FILLED_PAUSE_RE = re.compile(
    r"^(э+|э?м+|а{2,}|у{2,}|о{2,}|ы+)$"
    r"|^[аэоуым]([-–][аэоуым]+)+$"
)
FRAGMENT_RE = re.compile(r"[-–]{1,2}$")


def norm(token: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]", "", token.lower()).replace("ё", "е")


def parse_transcript(path: Path):
    segs, cur = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            cur = {"start": int(m.group(1)) * 60 + int(m.group(2)),
                   "end": int(m.group(3)) * 60 + int(m.group(4)),
                   "speaker": m.group(5), "lines": []}
            segs.append(cur)
        elif line.strip() and cur is not None:
            cur["lines"].append(line.strip())
    for seg in segs:
        seg["text"] = " ".join(seg.pop("lines"))
    return segs


def parse_words(path: Path):
    words = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            continue
        # В ранних выгрузках поле называлось word — принимаем оба варианта.
        out.append({"text": w.get("text") or w.get("word") or "",
                    "start": w["start"], "end": w["end"],
                    "speaker": w.get("speaker")})
    return out


def classify_hesitation(prev_text: str, text: str, min_repeat_len: int = 2):
    """Тип запинки по форме слова. Причины не угадываем."""
    clean = re.sub(r"[^0-9a-zа-я-]", "", text.strip().lower().replace("ё", "е"))
    if not clean:
        return None
    if FRAGMENT_RE.search(clean):
        return "обрыв слова"
    if FILLED_PAUSE_RE.match(clean):
        return "заполненная пауза"
    if norm(prev_text) and norm(prev_text) == norm(text) and len(norm(text)) >= min_repeat_len:
        return "повтор слова"
    return None


def count_parasites(words, singles, pairs) -> int:
    n = sum(1 for w in words if w in singles)
    n += sum(1 for a, b in zip(words, words[1:]) if [a, b] in pairs)
    return n


def per_speaker(segments, names, cfg):
    people = {}
    for seg in segments:
        spk = names.get(seg["speaker"], seg["speaker"])
        p = people.setdefault(spk, {"turns": [], "words": 0, "parasites": 0,
                                    "questions": 0, "speaking": 0.0, "rates": []})
        dur = max(seg["end"] - seg["start"], 0.001)
        ws = [w for w in (norm(t) for t in seg["text"].split()) if w]
        p["turns"].append(dur)
        p["words"] += len(ws)
        p["parasites"] += count_parasites(ws, set(cfg["parasites"]), cfg["parasite_pairs"])
        p["questions"] += seg["text"].count("?")
        p["speaking"] += dur
        if dur >= cfg["min_turn_for_rate"] and ws:
            p["rates"].append(len(ws) / dur * 60)
    for p in people.values():
        p["rate"] = st.median(p["rates"]) if p["rates"] else 0
        p["rate_spread"] = st.pstdev(p["rates"]) if len(p["rates"]) > 1 else 0
        p["turn_mean"] = st.mean(p["turns"]) if p["turns"] else 0
        p["turn_max"] = max(p["turns"]) if p["turns"] else 0
    return people


def transitions(segments, names):
    pauses, overlaps = [], {}
    for prev, cur in zip(segments, segments[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        gap = cur["start"] - prev["end"]
        if gap >= 0:
            pauses.append(gap)
        else:
            who = names.get(cur["speaker"], cur["speaker"])
            overlaps[who] = overlaps.get(who, 0) + 1
    return pauses, overlaps


def inside_turns(words, names, cfg):
    """Паузы и запинки внутри речи одного человека."""
    per, prev, prev_text = {}, None, ""
    for w in words:
        spk = names.get(w.get("speaker"), w.get("speaker") or "—")
        p = per.setdefault(spk, {"pauses": [], "hesitations": {}, "words": 0})
        p["words"] += 1
        kind = classify_hesitation(prev_text, w["text"])
        if kind:
            p["hesitations"][kind] = p["hesitations"].get(kind, 0) + 1
        # Промежуток между репликами разных людей — смена говорящего,
        # а не заминка, поэтому считаем только внутри одного голоса.
        if prev is not None and prev.get("speaker") == w.get("speaker"):
            gap = w["start"] - prev["end"]
            if 0 < gap < 30:
                p["pauses"].append(gap)
        prev, prev_text = w, w["text"]
    return per
