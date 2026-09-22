"""Реэкспорт модуля объективного анализа речи."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import speech_analysis as _sa
from speech_analysis import acoustics, config, metrics, report, brief, llm, analyze

__all__ = ["analyze", "acoustics", "config", "metrics", "report", "brief", "llm"]
