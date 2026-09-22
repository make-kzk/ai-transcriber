#!/usr/bin/env python3
"""Из чего состоят расхождения между двумя расшифровками.

Воспроизводит разбор из README: какая доля различий приходится на
запинки, слова-паразиты, латиницу против кириллицы и настоящий разный
текст. Показывает, что две трети расхождений между системами — вопрос
стиля, а не точности.

    python measurements/divergence_categories.py основа.txt сверка.txt
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_transcripts as ct

LATIN = re.compile(r"[a-z]")
DISFLUENCY = re.compile(r"(.)\1|--|^[аэмну]+$")


def categorise(left, right) -> str:
    if ct.as_number(left) is not None and ct.as_number(right) is not None:
        return "числа"
    if bool(any(LATIN.search(w) for w in left)) != bool(any(LATIN.search(w) for w in right)):
        return "латиница против кириллицы"
    if all(w in ct.FILLER for w in left + right):
        return "слова-паразиты"
    if any(DISFLUENCY.search(w) for w in left + right):
        return "запинки и междометия"
    if not left:
        return "есть только во второй"
    if not right:
        return "есть только в первой"
    return "разный текст"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--examples", type=int, default=6, help="сколько примеров на категорию")
    args = ap.parse_args()

    a, b = ct.parse(args.base), ct.parse(args.other)
    if len(a) != len(b):
        sys.exit(f"Разное число реплик: {len(a)} и {len(b)}")

    groups = {}
    for sa, sb in zip(a, b):
        ta, tb = sa["text"].split(), sb["text"].split()
        na, nb = [ct.norm(t) for t in ta], [ct.norm(t) for t in tb]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb).get_opcodes():
            if tag == "equal":
                continue
            left = [w for w in na[i1:i2] if w]
            right = [w for w in nb[j1:j2] if w]
            if not left and not right:
                continue
            groups.setdefault(categorise(left, right), []).append(
                (sa["start"], " ".join(ta[i1:i2]) or "—", " ".join(tb[j1:j2]) or "—"))

    total = sum(len(v) for v in groups.values())
    print(f"Всего расхождений: {total}\n")
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:28} {len(items):4}  ({len(items)/total*100:.0f}%)")
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n— {name}")
        shown = 0
        for start, left, right in items:
            if len(left) < 42 and len(right) < 42:
                print(f"    [{start}] «{left}» → «{right}»")
                shown += 1
            if shown >= args.examples:
                break


if __name__ == "__main__":
    main()
