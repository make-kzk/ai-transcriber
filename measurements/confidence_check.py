#!/usr/bin/env python3
"""Проверка гипотезы: годятся ли оценки выравнивания как детектор ошибок.

Гипотеза была такая: wav2vec2 при выравнивании даёт оценку на каждое
слово, значит низкая оценка укажет на ошибку распознавания — и вторая
система не понадобится.

Проверка её опровергла, и этот скрипт показывает почему. Оценка меряет
длину слова, а не правильность, и слепа к пропущенным словам: оценивать
нечего, если слова нет.

Нужен файл пословных таймкодов, содержащий поле score (его даёт
выравнивание whisperx).

    python measurements/confidence_check.py v3.txt v2.txt v3_слова.json
"""
import argparse
import difflib
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_transcripts as ct


def text_words(path: Path):
    text = re.sub(r"^\[\d\d:\d\d - \d\d:\d\d\] SPEAKER_\d+:$", " ",
                  path.read_text(encoding="utf-8"), flags=re.M)
    return [w for w in (ct.norm(t) for t in text.split()) if w]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="расшифровка, к которой относятся оценки")
    ap.add_argument("other", type=Path, help="вторая расшифровка — источник истины о расхождениях")
    ap.add_argument("words", type=Path, help="пословные таймкоды с полем score")
    args = ap.parse_args()

    raw = json.loads(args.words.read_text(encoding="utf-8"))
    scored = [(ct.norm(w.get("text") or w.get("word") or ""), w["score"])
              for w in raw if w.get("score") is not None]
    scored = [(t, s) for t, s in scored if t]
    if not scored:
        sys.exit("В файле нет поля score — эта проверка требует выгрузки с оценками.")

    wb, wo = text_words(args.base), text_words(args.other)

    # Сопоставляем последовательность из JSON с текстом расшифровки.
    index = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, [t for t, _ in scored], wb).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                index[j1 + k] = scored[i1 + k][1]

    changed = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wo, wb).get_opcodes():
        if tag != "equal":
            changed.update(range(j1, j2))

    diff = [index[i] for i in changed if i in index]
    same = [s for i, s in index.items() if i not in changed]
    print(f"сопоставлено слов: {len(index)} из {len(wb)}")
    print(f"{'':>12} {'слов':>6} {'медиана':>9} {'доля ниже 0.5':>15}")
    for name, values in (("разошлись", diff), ("совпали", same)):
        if values:
            low = sum(1 for v in values if v < 0.5) / len(values) * 100
            print(f"{name:>12} {len(values):>6} {st.median(values):>9.3f} {low:>14.0f}%")

    print("\nсредняя оценка по длине слова:")
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 9), (10, 30)):
        vals = [s for t, s in scored if lo <= len(t) <= hi]
        if vals:
            print(f"  {lo}-{hi} букв: {st.mean(vals):.3f}  ({len(vals)} слов)")
    print("\nЕсли средняя оценка растёт с длиной слова, а разница между")
    print("разошедшимися и совпавшими мала — оценка меряет длину, не правильность.")


if __name__ == "__main__":
    main()
