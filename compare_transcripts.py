#!/usr/bin/env python3
"""Сведение двух расшифровок одной записи с пометкой смысловых расхождений.

Фасад для обратной совместимости: делегирует логику в ai_transcriber.core.compare.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ai_transcriber.core.compare import (
    main,
    compare_files,
    classify,
    has_negation,
    norm,
    as_number,
    word_number,
    drop_filler_phrases,
    split_tokens,
    parse,
    moved_to_neighbour,
    compare,
    render_header,
    NEGATION,
    NEGATION_RE,
    SAME_WORD,
    FILLER_PHRASES,
    FILLER,
    NUM_EXACT,
    NUM_PREFIX,
    SEG_RE,
)

if __name__ == "__main__":
    sys.exit(main())
