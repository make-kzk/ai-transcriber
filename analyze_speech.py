#!/usr/bin/env python3
"""Объективные метрики речи из готовой расшифровки.

Считает только то, что измеримо и проверяемо: темп, баланс времени,
паузы, перебивания, слова-паразиты, долю вопросов. Никаких ярлыков
вроде «уверенно» или «радость» — такие оценки по тексту и таймкодам
не выводятся, а выглядят убедительно и потому вводят в заблуждение.

Работает с любой расшифровкой в нашем формате, независимо от того,
какая система её сделала.
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcript_utils as tu
from compare_transcripts import norm


# Свой список, а не FILLER из модуля сверки: там в паразиты записаны «и»,
# «а», «как», «то» — для сравнения текстов их пропажа действительно шум, но
# это обычные служебные слова, и считать их за паразиты речи нельзя.
PARASITES = {"ну", "вот", "типа", "короче", "блин", "значит", "слушай",
             "прям", "прямо", "походу", "реально", "просто", "получается"}
PARASITE_PAIRS = {("как", "бы"), ("то", "есть"), ("в", "общем"),
                  ("это", "самое"), ("так", "сказать"), ("на", "самом")}


def count_parasites(words) -> int:
    """Одиночные слова плюс устойчивые обороты вроде «как бы»."""
    n = sum(1 for w in words if w in PARASITES)
    n += sum(1 for a, b in zip(words, words[1:]) if (a, b) in PARASITE_PAIRS)
    return n


# Запинки видны только там, где система их сохраняет. Whisper причёсывает
# речь и выбрасывает «э-э» и обрывы слов, ElevenLabs и Deepgram оставляют.
# Поэтому счёт запинок осмысленно сравнивать лишь внутри одной системы.
# Только то, что не является нормальным словом: тянущийся гласный, «м»,
# или слог через дефис. Набор букв без этого условия ловил «а», «у», «на»,
# «ну» — и давал у Whisper сотню несуществующих запинок.
HESITATION_RE = re.compile(
    r"^(э+|э?м+|а{2,}|у{2,}|о{2,}|ы+)$"      # э, ээ, эм, мм, ааа, ууу
    r"|^[аэоуым]([-–][аэоуым]+)+$"           # а-а, э-э-э, м-м
)
FRAGMENT_RE = re.compile(r"[-–]{1,2}$")                            # оцен--, кон-


def classify_hesitation(prev_word: str, word: str) -> str | None:
    """Тип запинки или None. Только по форме слова, без догадок о причинах."""
    bare = word.strip().lower().replace("ё", "е")
    clean = re.sub(r"[^0-9a-zа-я-]", "", bare)
    if not clean:
        return None
    if FRAGMENT_RE.search(clean):
        return "обрыв слова"
    if HESITATION_RE.match(clean):
        return "заполненная пауза"
    if prev_word and re.sub(r"[^0-9a-zа-я]", "", prev_word.lower()) == \
            re.sub(r"[^0-9a-zа-я]", "", bare) and len(clean) > 1:
        return "повтор слова"
    return None


def analyze_words(path: Path, names):
    """Паузы и запинки внутри реплик — по пословным таймкодам."""
    words = json.loads(path.read_text(encoding="utf-8"))
    per = {}
    prev, prev_text = None, ""
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            continue
        spk = names.get(w.get("speaker"), w.get("speaker") or "—")
        p = per.setdefault(spk, {"pauses": [], "hesitations": {}, "words": 0})
        p["words"] += 1
        # В ранних выгрузках поле называлось word — принимаем оба варианта.
        text = w.get("text") or w.get("word") or ""
        kind = classify_hesitation(prev_text, text)
        if kind:
            p["hesitations"][kind] = p["hesitations"].get(kind, 0) + 1
        # Промежуток считаем только внутри речи одного человека: пауза
        # между репликами разных людей — это смена говорящего, а не заминка.
        if prev is not None and prev.get("speaker") == w.get("speaker"):
            gap = w["start"] - prev["end"]
            if 0 < gap < 30:
                p["pauses"].append(gap)
        prev, prev_text = w, text
    return per


def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def words_of(text: str):
    return [w for w in (norm(t) for t in text.split()) if w]


def collect(segments, names):
    """Группирует реплики по говорящим и считает базовые величины."""
    people = {}
    for i, seg in enumerate(segments):
        spk = names.get(seg["speaker"], seg["speaker"])
        p = people.setdefault(spk, {
            "turns": [], "words": 0, "fillers": 0, "questions": 0,
            "speaking": 0.0, "rates": [],
        })
        dur = max(seg["end"] - seg["start"], 0.001)
        ws = words_of(seg["text"])
        p["turns"].append(dur)
        p["words"] += len(ws)
        p["fillers"] += count_parasites(ws)
        p["questions"] += seg["text"].count("?")
        p["speaking"] += dur
        # Темп меряем только на репликах длиннее пяти секунд: на коротких
        # округление таймкодов до секунды даёт дикий разброс.
        if dur >= 5 and ws:
            p["rates"].append(len(ws) / dur * 60)
    return people


def transitions(segments, names):
    """Паузы и перебивания между соседними репликами разных людей."""
    pauses, overlaps = [], {}
    for prev, cur in zip(segments, segments[1:]):
        gap = cur["start"] - prev["end"]
        if prev["speaker"] == cur["speaker"]:
            continue
        if gap >= 0:
            pauses.append(gap)
        else:
            who = names.get(cur["speaker"], cur["speaker"])
            overlaps[who] = overlaps.get(who, 0) + 1
    return pauses, overlaps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--names", help="имена вместо SPEAKER_00, через запятую")
    ap.add_argument("--words", type=Path,
                    help="файл с пословными таймкодами (по умолчанию ищется рядом)")
    ap.add_argument("-o", "--output", type=Path, help="сохранить отчёт в файл")
    args = ap.parse_args()

    if not args.transcript.exists():
        sys.exit(f"Файл не найден: {args.transcript}")
    segments = tu.parse_transcript(args.transcript)
    if not segments:
        sys.exit("В файле не найдено ни одной реплики — это точно расшифровка?")

    names = {}
    if args.names:
        for i, name in enumerate(n.strip() for n in args.names.split(",")):
            names[f"SPEAKER_{i:02d}"] = name

    people = collect(segments, names)
    pauses, overlaps = transitions(segments, names)
    total_speech = sum(p["speaking"] for p in people.values())
    span = segments[-1]["end"] - segments[0]["start"]

    out = []
    add = out.append
    add("=" * 62)
    add(f"РЕЧЕВЫЕ МЕТРИКИ: {args.transcript.name}")
    add("=" * 62)
    add(f"Длительность: {mmss(span)}, реплик: {len(segments)}, "
        f"говорящих: {len(people)}")
    add("")

    for spk, p in sorted(people.items(), key=lambda kv: -kv[1]["speaking"]):
        share = p["speaking"] / total_speech * 100 if total_speech else 0
        rate = st.median(p["rates"]) if p["rates"] else 0
        spread = st.pstdev(p["rates"]) if len(p["rates"]) > 1 else 0
        filler = p["fillers"] / p["words"] * 100 if p["words"] else 0
        add(f"── {spk}")
        add(f"   Времени в эфире: {mmss(p['speaking'])} ({share:.0f}%)")
        add(f"   Реплик: {len(p['turns'])}, слов: {p['words']}")
        add(f"   Средняя реплика: {st.mean(p['turns']):.0f} с, "
            f"самая длинная: {max(p['turns']):.0f} с")
        add(f"   Темп речи: {rate:.0f} слов/мин (разброс ±{spread:.0f})")
        add(f"   Слова-паразиты: {filler:.1f}% речи ({p['fillers']} шт.)")
        add(f"   Вопросов задано: {p['questions']}")
        if overlaps.get(spk):
            add(f"   Вступал поверх собеседника: {overlaps[spk]} раз")
        add("")

    words_file = args.words or args.transcript.with_name(
        args.transcript.stem + "_слова.json")
    if words_file.exists():
        inner = analyze_words(words_file, names)
        add("── Внутри реплик")
        for spk, d in sorted(inner.items(), key=lambda kv: -kv[1]["words"]):
            if not d["words"]:
                continue
            long_p = [g for g in d["pauses"] if g >= 0.5]
            very = [g for g in d["pauses"] if g >= 1.5]
            add(f"   {spk}:")
            add(f"      Пауз дольше 0,5 с: {len(long_p)}"
                + (f" (из них дольше 1,5 с: {len(very)})" if very else "")
                + (f", самая длинная {max(d['pauses']):.1f} с" if d["pauses"] else ""))
            if d["pauses"]:
                add(f"      На 100 слов приходится пауз: "
                    f"{len(long_p) / d['words'] * 100:.1f}")
            total_h = sum(d["hesitations"].values())
            if total_h:
                parts = ", ".join(f"{k} — {v}" for k, v in
                                  sorted(d["hesitations"].items(), key=lambda kv: -kv[1]))
                add(f"      Запинок: {total_h} ({parts})")
                add(f"      На 100 слов: {total_h / d['words'] * 100:.1f}")
            else:
                add("      Запинок не найдено — возможно, система их вычищает.")
        add("")
    else:
        add("── Внутри реплик")
        add(f"   Нет файла {words_file.name} — паузы и запинки внутри реплик")
        add("   без пословных таймкодов не считаются.")
        add("")

    add("── Паузы между репликами")
    if pauses:
        long = [g for g in pauses if g >= 2]
        add(f"   Смен говорящего: {len(pauses)}, медиана паузы: "
            f"{st.median(pauses):.1f} с")
        add(f"   Пауз дольше 2 секунд: {len(long)}"
            + (f", самая длинная {max(pauses):.0f} с" if pauses else ""))
    else:
        add("   Смен говорящего не зафиксировано.")
    add("")
    add("Метрики объективны, но зависят от данных. Время реплик округлено")
    add("до секунды: темп на коротких репликах не считается. Запинки видны")
    add("только там, где система их сохраняет — Whisper причёсывает речь,")
    add("ElevenLabs и Deepgram оставляют, поэтому сравнивать их счёт между")
    add("системами бессмысленно.")
    add("=" * 62)

    report = "\n".join(out)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nСохранено: {args.output}")


if __name__ == "__main__":
    main()
