"""Базовый интерфейс для ASR-движков."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..core.models import EngineResult


class BaseTranscriber(ABC):
    """Абстрактный класс адаптера системы распознавания речи."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Внутренний идентификатор движка (whisper-v3, elevenlabs, gemini и т.д.)."""
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """Человекочитаемое название."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str = "ru",
        speakers: Optional[int] = None,
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> EngineResult:
        """Запускает распознавание и возвращает результат."""
        pass
