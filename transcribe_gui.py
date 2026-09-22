#!/usr/bin/env python3
"""Графическое приложение AI Transcriber на Tkinter.

Фасад для обратной совместимости: делегирует запуск в ai_transcriber.gui.app.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ai_transcriber.gui.app import main, TranscribeApp

if __name__ == "__main__":
    main()
