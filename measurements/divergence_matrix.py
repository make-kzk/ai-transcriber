#!/usr/bin/env python3
"""Попарное расхождение по словам между расшифровками.

Воспроизводит таблицу из README, которой обосновано, что ошибки систем
коррелируют: Whisper, Nexara и Deepgram ближе друг к другу, чем к
ElevenLabs, и потому сверять их между собой малополезно.

    python measurements/divergence_matrix.py "Whisper=v3.txt" "ElevenLabs=el.txt" ...

Имена до знака «равно» — подписи для таблицы. Файлы должны быть разбиты
на одинаковые реплики (внешние системы приводятся к разбивке эталона
адаптерами).
"""
import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_transcripts as ct


def words(segment):
    return [w for w in (ct.norm(t) for t in segment["text"].split()) if w]


def divergence(a, b) -> float:
    diff = total = 0
    for sa, sb in zip(a, b):
        wa, wb = words(sa), words(sb)
        total += max(len(wa), len(wb))
        matcher = difflib.SequenceMatcher(None, wa, wb)
        diff += sum(max(i2 - i1, j2 - j1)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return diff / total * 100 if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", metavar="ПОДПИСЬ=файл")
    args = ap.parse_args()

    named = {}
    for item in args.files:
        label, _, path = item.partition("=")
        path = Path(path or label)
        if not path.exists():
            sys.exit(f"Файл не найден: {path}")
        named[label if path.name != label else path.stem[:20]] = ct.parse(path)

    lengths = {k: len(v) for k, v in named.items()}
    if len(set(lengths.values())) > 1:
        sys.exit(f"Разное число реплик — построчно сравнить нельзя: {lengths}")

    print(f"Реплик в каждом файле: {next(iter(lengths.values()))}\n")
    for name, segs in named.items():
        print(f"  {name:22} слов: {sum(len(words(s)) for s in segs)}")
    print("\nРасхождение по словам:")
    names = list(named)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((divergence(named[names[i]], named[names[j]]),
                          names[i], names[j]))
    for value, a, b in sorted(pairs):
        print(f"  {a:18} ↔ {b:18} {value:5.1f}%")


if __name__ == "__main__":
    main()
