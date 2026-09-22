"""Адаптер к Google Gemini 3.5 Transcribe."""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core import utils
from ..core.models import EngineResult
from .base import BaseTranscriber

MODEL = "gemini-3.5-transcribe"

DEFAULT_VOCABULARY = [
    "CEO", "CFO", "CMO", "CPO", "CTO", "COO", "C-level",
    "HR", "HRD", "hiring manager", "job offer", "offer",
    "iGaming", "fintech", "blockchain", "e-com", "DevOps",
    "backend", "frontend", "R&D", "LinkedIn", "onboarding",
    "welcome-встреча", "релокация", "испытательный срок",
]


def _seconds(value) -> Optional[float]:
    """Смещения приходят числом или строкой вида '1.5s'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("s")
    try:
        return float(text)
    except ValueError:
        return None


def extract_words(interaction) -> list[dict]:
    """Слова с таймкодами и спикером из ответа Gemini (поддерживает объекты SDK и dict)."""
    words = []
    # Если передан словарь (например, в тестах)
    if isinstance(interaction, dict):
        steps = interaction.get("steps", [])
        for step in steps:
            for content in step.get("content", []):
                for ann in content.get("annotations", []):
                    if ann.get("type") != "word_info":
                        continue
                    start = _seconds(ann.get("start_offset"))
                    end = _seconds(ann.get("end_offset"))
                    if start is None:
                        continue
                    words.append({
                        "text": ann.get("text", ""),
                        "start": start,
                        "end": end if end is not None else start,
                        "speaker": ann.get("speaker"),
                    })
        return words

    # Если передан объект Google GenAI SDK
    for step in getattr(interaction, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                if getattr(ann, "type", None) != "word_info":
                    continue
                start = _seconds(getattr(ann, "start_offset", None))
                end = _seconds(getattr(ann, "end_offset", None))
                if start is None:
                    continue
                words.append({
                    "text": getattr(ann, "text", ""),
                    "start": start,
                    "end": end if end is not None else start,
                    "speaker": getattr(ann, "speaker", None),
                })
    return words


def recognize(audio: Path, language: str, vocabulary: Optional[list] = None,
              model: str = MODEL, on_progress=None):
    """Отправка аудио в Gemini API через google-genai SDK."""
    from google import genai
    from google.genai import types

    def say(text):
        print(text, flush=True)
        if on_progress:
            on_progress(text)

    client = genai.Client()
    say(f"🚀 Загрузка в Gemini: {audio.name} ({audio.stat().st_size / (1024*1024):.1f} МБ)")
    uploaded = client.files.upload(file=str(audio))

    deadline = time.time() + 900
    while uploaded.state == types.FileState.PROCESSING and time.time() < deadline:
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Запись не принята сервисом: состояние {uploaded.state}")

    config = {"language_codes": [language]}
    if vocabulary:
        say(f"   Словарь-подсказка: {len(vocabulary)} терминов (таймкоды при этом недоступны)")
        config["custom_vocabulary"] = vocabulary
        config["mode"] = {"type": "verbatim", "diarization_mode": "speaker"}
    else:
        say("   Пословные таймкоды и диаризация")
        config["mode"] = {"type": "verbatim", "diarization_mode": "speaker",
                          "timestamp_granularities": ["word"]}

    return client.interactions.create(
        model=model,
        input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
        generation_config={"transcription_config": config},
    )


class GeminiEngine(BaseTranscriber):
    name = "gemini"
    label = "Gemini 3.5 Transcribe"

    def __init__(self, model: str = MODEL, vocabulary: Optional[list] = None):
        self.model = model
        self.vocabulary = vocabulary

    def transcribe(
        self,
        audio_path: Path,
        language: str = "ru-RU",
        speakers: Optional[int] = None,
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> EngineResult:
        if dry_run:
            out_dir = output_dir or audio_path.parent
            stamp = "<dry-run>"
            base = out_dir / f"{audio_path.stem}_gemini_{stamp}.txt"
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=True,
                transcript_path=base,
                words_path=out_dir / f"{audio_path.stem}_gemini_{stamp}_слова.json",
            )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=False,
                error="Не задан GEMINI_API_KEY или GOOGLE_API_KEY",
            )

        try:
            lang = "ru-RU" if language in ("ru", "rus", "ru-RU") else ("en-US" if language in ("en", "eng", "en-US") else language)
            interaction = recognize(audio_path, language=lang, vocabulary=self.vocabulary, model=self.model)
            words = extract_words(interaction)

            out_dir = output_dir or audio_path.parent
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            if not words:
                text = (getattr(interaction, "output_text", "") or "").strip()
                out = out_dir / f"{audio_path.stem}_gemini-словарь_{stamp}.txt"
                out.write_text(text, encoding="utf-8")
                return EngineResult(
                    engine_key=self.name,
                    label=self.label,
                    success=True,
                    transcript_path=out,
                )

            speaker_of = utils.SpeakerMap()
            base = out_dir / f"{audio_path.stem}_gemini_{stamp}.txt"
            base.write_text(utils.native_transcript(words, speaker_of), encoding="utf-8")
            words_file = utils.save_words(words, base)

            fitted_file = None
            if reference_path and reference_path.exists():
                ref_segs = utils.parse_reference(reference_path)
                if ref_segs:
                    fitted_file = out_dir / f"{audio_path.stem}_gemini_свод_{stamp}.txt"
                    fitted_file.write_text(utils.fitted_transcript(words, ref_segs), encoding="utf-8")

            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=True,
                transcript_path=base,
                words_path=words_file,
                fitted_path=fitted_file,
            )
        except Exception as e:
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=False,
                error=str(e),
            )
