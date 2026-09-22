"""Общие утилиты транскрибации.

Фасад для обратной совместимости: делегирует функции в ai_transcriber.core.utils.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ai_transcriber.core.utils import (
    mmss,
    SpeakerMap,
    native_transcript,
    parse_reference,
    save_words,
    load_words,
    parse_transcript,
    fitted_transcript,
    assign_words_to_segments,
    format_transcript,
    clean_line,
    SEG_RE,
)
from ai_transcriber.core.compare import compare_files


def run_comparison(base: Path, other: Path, suffix: str):
    """Сводит расшифровки и помечает расхождения."""
    try:
        _, findings, out = compare_files(base, other)
        crit = sum(1 for f in findings if f["level"] == "critical")
        cont = sum(1 for f in findings if f["level"] == "content")
        print("\n🔍 Сверка с независимой системой:")
        print(f"   критичных расхождений: {crit}")
        print(f"   смысловых разночтений: {cont}")
        print(f"📋 {out.name}")
    except Exception as e:
        print(f"⚠️ Не удалось выполнить сверку: {e}")
