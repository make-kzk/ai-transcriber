"""Ядро ai_transcriber: модели, утилиты, сравнение и оркестрация пайплайна."""
from .models import Word, Segment, DiffItem, DiffSummary, EngineResult
from .utils import (
    load_words,
    save_words,
    parse_reference,
    assign_words_to_segments,
    fitted_transcript,
    format_transcript,
    clean_line,
)
from .compare import compare_files, classify, DivergenceClassifier
from .pipeline import run_pipeline, PIPELINE_ENGINES

__all__ = [
    "Word",
    "Segment",
    "DiffItem",
    "DiffSummary",
    "EngineResult",
    "load_words",
    "save_words",
    "parse_reference",
    "assign_words_to_segments",
    "fitted_transcript",
    "format_transcript",
    "clean_line",
    "compare_files",
    "classify",
    "DivergenceClassifier",
    "run_pipeline",
    "PIPELINE_ENGINES",
]
