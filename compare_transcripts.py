#!/usr/bin/env python3
"""Сведение двух расшифровок одной записи с пометкой смысловых расхождений.

Две модели, независимо распознавшие одно аудио, расходятся примерно в 6%
слов. Подавляющая часть этих расхождений — пунктуация, регистр и слова-
паразиты, читать их бессмысленно. Опасны единицы: потерянное отрицание или
изменённое число меняют смысл, а текст при этом выглядит гладким.

Скрипт берёт за основу одну расшифровку, помечает в ней места несогласия
и выносит критичные наверх — чтобы их можно было переслушать по таймкоду.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

SEG_RE = re.compile(r"^\[(\d\d):(\d\d) - (\d\d):(\d\d)\] (SPEAKER_\d+):$")

# Потеря или появление этих слов переворачивает смысл фразы.
NEGATION = {"не", "нет", "ни", "нельзя", "без"}
# Отрицательные местоимения склоняются: никакой, никакого, никем, ничему.
NEGATION_RE = re.compile(r"^(ника\w*|никог\w*|никт\w*|ником\w*|никем|нич\w*|нискол\w*)$")


def has_negation(words) -> bool:
    return any(w in NEGATION or NEGATION_RE.match(w) for w in words)

# Разговорные варианты одного слова — расхождением не считаются.
SAME_WORD = {"нету": "нет", "нема": "нет", "ага": "да", "угу": "да"}

# Обороты, где «не» — часть слова-паразита, а не отрицание по смыслу.
FILLER_PHRASES = (("не", "знаю"), ("то", "есть"))

# Вставка или пропуск слова-паразита смысла не меняет.
FILLER = {
    "ну", "вот", "да", "там", "то", "есть", "как", "бы", "угу", "это", "а", "и",
    "же", "ж", "типа", "короче", "просто", "значит", "так", "окей", "уже", "вообще",
    "самом", "деле", "конечно", "прям", "прямо", "тоже", "еще", "ещё", "опять",
}

# Короткие числительные — только точным совпадением: по префиксу «сто»
# поймало бы «стоит», «три» — «тридцать».
NUM_EXACT = {
    "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3, "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "сто": 100, "ста": 100, "сорок": 40, "сорока": 40,
}

# Длинные основы — по началу слова: числительные склоняются, и перечислять
# все формы («двадцать», «двадцати», «двадцатью») бессмысленно.
# Порядок важен: длинные основы проверяются первыми, иначе «три» съест
# «тридцать».
NUM_PREFIX = sorted([
    ("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14),
    ("пятнадцат", 15), ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18),
    ("девятнадцат", 19), ("двадцат", 20), ("тридцат", 30),
    ("пятьдесят", 50), ("пятидесят", 50), ("шестьдесят", 60), ("шестидесят", 60),
    ("семьдесят", 70), ("семидесят", 70), ("восемьдесят", 80), ("восьмидесят", 80),
    ("девяност", 90), ("двест", 200), ("трист", 300), ("четырест", 400),
    ("пятьсот", 500), ("пятист", 500), ("шестьсот", 600), ("семьсот", 700),
    ("восемьсот", 800), ("девятьсот", 900),
    ("тысяч", 1000), ("миллион", 1000000), ("миллиард", 1000000000),
], key=lambda kv: -len(kv[0]))


def word_number(word: str):
    """Число, записанное словом, — или None."""
    if word in NUM_EXACT:
        return NUM_EXACT[word]
    for stem, value in NUM_PREFIX:
        if word.startswith(stem):
            return value
    return None


def norm(token: str) -> str:
    """Слово без пунктуации и регистра — для сравнения, не для вывода."""
    w = re.sub(r"[^0-9a-zа-яё]", "", token.lower()).replace("ё", "е")
    return SAME_WORD.get(w, w)


def as_number(tokens) -> int | None:
    """«2 тысячи» и «2000» — одно и то же число, записанное по-разному."""
    total, seen = 0, False
    for t in tokens:
        if t.isdigit():
            total = total * 1000 if t == "000" else total + int(t)
            seen = True
            continue
        value = word_number(t)
        if value is None:
            return None
        # «2 тысячи» — умножение, «двадцать пять» — сложение.
        total = total * value if value >= 1000 and total else total + value
        seen = True
    return total if seen else None


def drop_filler_phrases(words):
    """Убирает «не знаю» и подобное — там «не» не отрицание, а часть оборота."""
    out, i = [], 0
    while i < len(words):
        for phrase in FILLER_PHRASES:
            if tuple(words[i:i + len(phrase)]) == phrase:
                i += len(phrase)
                break
        else:
            out.append(words[i])
            i += 1
    return out


def split_tokens(raw_tokens):
    """«3-5,» -> ['3', '5']: иначе диапазон не сравнить с «три, пять»."""
    out = []
    for t in raw_tokens:
        out.extend(w for w in re.split(r"[^0-9a-zа-яё]+", t.lower().replace("ё", "е")) if w)
    return [SAME_WORD.get(w, w) for w in out]


def classify(left_raw, right_raw):
    """Насколько расхождение важно: critical / content / noise."""
    left = drop_filler_phrases(split_tokens(left_raw))
    right = drop_filler_phrases(split_tokens(right_raw))
    ls, rs = set(left), set(right)

    if not left and not right:
        return "noise"

    # Критично, когда отрицание есть с одной стороны и пропало с другой.
    # Если оно есть в обеих — это перестройка фразы, смысл сохранён:
    # «никакой замене» и «ни о какой замене» значат одно и то же.
    if has_negation(left) != has_negation(right):
        return "critical"

    ln, rn = as_number(left), as_number(right)
    if ln is not None and rn is not None:
        return "noise" if ln == rn else "critical"
    if (ln is None) != (rn is None) and (left and right):
        return "content"

    meaningful = lambda ws: {w for w in ws if w not in FILLER and len(w) > 2}
    if not meaningful(ls) and not meaningful(rs):
        return "noise"
    return "content"


def parse(path: Path):
    segments, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            current = {
                "head": line.strip(),
                "start": f"{m.group(1)}:{m.group(2)}",
                "speaker": m.group(5),
                "lines": [],
            }
            segments.append(current)
        elif line.strip() and current is not None:
            current["lines"].append(line.strip())
    for s in segments:
        s["text"] = " ".join(s["lines"])
    return segments


def moved_to_neighbour(words, segments, index) -> bool:
    """Текст не пропал, а уехал в соседнюю реплику — границы режутся по-разному."""
    # Короткий фрагмент соседом не объяснить: одиночное «не» есть в любой
    # реплике рядом, и такое правило съело бы ровно то, что мы ищем.
    if len(words) < 3:
        return False
    near = set()
    for j in (index - 1, index + 1):
        if 0 <= j < len(segments):
            near.update(norm(t) for t in segments[j]["text"].split())
    return sum(1 for w in words if w in near) >= max(1, len(words) * 0.6)


def compare(base_segs, other_segs, show_noise=False):
    """Возвращает текст сведённой расшифровки и список находок."""
    findings, body = [], []

    for index, (base, other) in enumerate(zip(base_segs, other_segs)):
        bt, ot = base["text"].split(), other["text"].split()
        bn, on = [norm(t) for t in bt], [norm(t) for t in ot]
        matcher = difflib.SequenceMatcher(None, bn, on)

        out, last = [], 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            level = classify(bt[i1:i2], ot[j1:j2])
            if level == "critical":
                # Пропажа с одной стороны часто означает лишь другую нарезку реплик.
                if not bn[i1:i2] and moved_to_neighbour(on[j1:j2], base_segs, index):
                    level = "noise"
                elif not on[j1:j2] and moved_to_neighbour(bn[i1:i2], other_segs, index):
                    level = "noise"
            if level == "noise" and not show_noise:
                continue
            mark = "!!" if level == "critical" else "??"
            left = " ".join(bt[i1:i2]) or "—"
            right = " ".join(ot[j1:j2]) or "—"
            out.extend(bt[last:i1])
            out.append(f"{mark}[{left} | {right}]{mark}")
            last = i2
            findings.append({
                "time": base["start"], "speaker": base["speaker"],
                "level": level, "left": left, "right": right,
                "context": " ".join(bt[max(0, i1 - 6):i2 + 6]),
            })
        out.extend(bt[last:])

        body.append(base["head"])
        body.append(" ".join(out))
        body.append("")

    return "\n".join(body), findings


def render_header(findings, base_name, other_name, segments):
    crit = [f for f in findings if f["level"] == "critical"]
    cont = [f for f in findings if f["level"] == "content"]
    lines = [
        "=" * 70,
        "СВЕДЁННАЯ РАСШИФРОВКА С ПОМЕТКАМИ РАСХОЖДЕНИЙ",
        "=" * 70,
        f"Основа:   {base_name}",
        f"Сверка с: {other_name}",
        f"Реплик: {len(segments)}. Расхождений: {len(crit)} критичных, {len(cont)} смысловых.",
        "",
        "Формат пометки:  !![основа | сверка]!!  — критично, переслушать",
        "                 ??[основа | сверка]??  — разночтение по смыслу",
        "",
    ]
    if crit:
        lines += ["-" * 70, "ПЕРЕСЛУШАТЬ В ПЕРВУЮ ОЧЕРЕДЬ", "-" * 70]
        for f in crit:
            lines += [
                f"[{f['time']}] {f['speaker']}:  «{f['left']}»  против  «{f['right']}»",
                f"          ...{f['context']}...",
                "",
            ]
    else:
        lines += ["Критичных расхождений не найдено.", ""]
    lines += ["=" * 70, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="расшифровка, берётся за основу")
    ap.add_argument("other", type=Path, help="вторая расшифровка, для сверки")
    ap.add_argument("-o", "--output", type=Path, help="куда записать (по умолчанию рядом с основой)")
    ap.add_argument("--show-noise", action="store_true",
                    help="помечать также пунктуацию, числа и слова-паразиты")
    args = ap.parse_args()

    for p in (args.base, args.other):
        if not p.exists():
            sys.exit(f"Файл не найден: {p}")

    base_segs, other_segs = parse(args.base), parse(args.other)
    if len(base_segs) != len(other_segs):
        sys.exit(
            f"Расшифровки разбиты на разное число реплик "
            f"({len(base_segs)} и {len(other_segs)}) — свести их построчно нельзя.\n"
            f"Так бывает, если файлы сделаны из разных записей."
        )

    body, findings = compare(base_segs, other_segs, args.show_noise)
    header = render_header(findings, args.base.name, args.other.name, base_segs)

    out = args.output or args.base.with_name(args.base.stem + "_сверка.txt")
    out.write_text(header + body, encoding="utf-8")

    crit = sum(1 for f in findings if f["level"] == "critical")
    cont = sum(1 for f in findings if f["level"] == "content")
    print(f"Сверено реплик: {len(base_segs)}")
    print(f"Критичных расхождений: {crit}")
    print(f"Смысловых разночтений: {cont}")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
