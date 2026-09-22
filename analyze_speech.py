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
    add("Метрики объективны, но зависят от точности таймкодов: они")
    add("округлены до секунды, поэтому темп на коротких репликах не")
    add("считается, а паузы короче секунды неразличимы.")
    add("=" * 62)

    report = "\n".join(out)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nСохранено: {args.output}")


if __name__ == "__main__":
    main()
