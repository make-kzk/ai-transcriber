"""Модуль ASR-движков ai_transcriber."""
from typing import Optional
from .base import BaseTranscriber
from .elevenlabs import ElevenLabsEngine
from .gemini import GeminiEngine
from .modal_whisper import ModalWhisperEngine

AVAILABLE_ENGINES = {
    "whisper-v3": ModalWhisperEngine,
    "whisper-turbo": ModalWhisperEngine,
    "whisper-v2": ModalWhisperEngine,
    "elevenlabs": ElevenLabsEngine,
    "gemini": GeminiEngine,
}


def get_engine(name: str, **kwargs) -> BaseTranscriber:
    """Фабрика для создания экземпляра движка по его ключу."""
    if name not in AVAILABLE_ENGINES:
        raise ValueError(
            f"Неизвестный или неподдерживаемый движок: '{name}'. "
            f"Доступные движки: {list(AVAILABLE_ENGINES.keys())}. "
            f"(Nexara и Deepgram признаны неэффективными и перемещены в deprecated/)"
        )
    cls = AVAILABLE_ENGINES[name]
    if cls is ModalWhisperEngine:
        return cls(key=name, **kwargs)
    return cls(**kwargs)


__all__ = [
    "BaseTranscriber",
    "ModalWhisperEngine",
    "ElevenLabsEngine",
    "GeminiEngine",
    "AVAILABLE_ENGINES",
    "get_engine",
]
