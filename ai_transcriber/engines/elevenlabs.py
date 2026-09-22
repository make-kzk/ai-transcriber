"""Адаптер к ElevenLabs Scribe STT API."""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from ..core import utils
from ..core.models import EngineResult
from .base import BaseTranscriber

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def extract_words(payload: dict) -> list[dict]:
    """Извлекает пословные таймкоды из ответа ElevenLabs Scribe."""
    return [
        {
            "text": str(w["text"]),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "speaker": w.get("speaker_id"),
        }
        for w in payload.get("words", [])
        if w.get("type") == "word" and w.get("start") is not None
    ]


words_of = extract_words


def recognize(
    audio: Path,
    api_key: str,
    language: str = "rus",
    speakers: Optional[int] = None,
    model: str = "scribe_v2",
    timeout: float = 900.0,
) -> dict:
    """Отправляет файл в ElevenLabs Scribe API."""
    data = {"model_id": model, "diarize": "true", "timestamps_granularity": "word"}
    if language and language != "auto":
        data["language_code"] = language
    if speakers:
        data["num_speakers"] = str(speakers)

    size_mb = audio.stat().st_size / (1024 * 1024)
    print(f"🚀 Отправка в ElevenLabs Scribe: {audio.name}")
    print(f"   Размер: {size_mb:.1f} МБ, модель: {model}, спикеров: {speakers or 'авто'}")

    with audio.open("rb") as fh:
        resp = httpx.post(
            API_URL,
            headers={"xi-api-key": api_key},
            data=data,
            files={"file": (audio.name, fh)},
            timeout=timeout,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs вернул {resp.status_code}: {resp.text[:500]}")
    return resp.json()


class ElevenLabsEngine(BaseTranscriber):
    name = "elevenlabs"
    label = "ElevenLabs Scribe"

    def __init__(self, api_key: Optional[str] = None, model: str = "scribe_v2"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.model = model

    def transcribe(
        self,
        audio_path: Path,
        language: str = "rus",
        speakers: Optional[int] = None,
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> EngineResult:
        if dry_run:
            out_dir = output_dir or audio_path.parent
            stamp = "<dry-run>"
            base = out_dir / f"{audio_path.stem}_elevenlabs_{stamp}.txt"
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=True,
                transcript_path=base,
                words_path=out_dir / f"{audio_path.stem}_elevenlabs_{stamp}_слова.json",
            )

        if not self.api_key:
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=False,
                error="Не задан ELEVENLABS_API_KEY",
            )

        try:
            lang = "rus" if language in ("ru", "rus") else ("eng" if language in ("en", "eng") else language)
            payload = recognize(
                audio_path,
                self.api_key,
                language=lang,
                speakers=speakers,
                model=self.model,
            )
            words = extract_words(payload)
            out_dir = output_dir or audio_path.parent
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            speaker_of = utils.SpeakerMap()
            base = out_dir / f"{audio_path.stem}_elevenlabs_{stamp}.txt"
            base.write_text(utils.native_transcript(words, speaker_of), encoding="utf-8")
            words_file = utils.save_words(words, base)

            fitted_file = None
            if reference_path and reference_path.exists():
                ref_segs = utils.parse_reference(reference_path)
                if ref_segs:
                    fitted_file = out_dir / f"{audio_path.stem}_elevenlabs_свод_{stamp}.txt"
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
