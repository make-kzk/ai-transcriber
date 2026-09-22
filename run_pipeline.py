#!/usr/bin/env python3
"""Прогоняет запись через выбранные системы распознавания и сводит результаты.

Фасад для обратной совместимости: делегирует логику в ai_transcriber.core.pipeline.
"""
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ai_transcriber.core.pipeline import (
    main,
    run_pipeline,
    PIPELINE_ENGINES as SYSTEMS,
    find_modal,
    run_parallel_jobs as run_parallel,
    fit_words_to_reference as fit_to_reference,
)

if __name__ == "__main__":
    sys.exit(main())
