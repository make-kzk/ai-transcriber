"""Адаптер для запуска моделей Whisper на Modal GPU."""
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.models import EngineResult
from .base import BaseTranscriber


def find_modal_bin() -> str:
    """Ищет утилиту modal в PATH и стандартных путях venv."""
    found = shutil.which("modal")
    if found:
        return found
    candidates = [
        Path.home() / ".local/bin/modal",
        Path.home() / ".modal-venv/bin/modal",
        Path("/usr/local/bin/modal"),
        Path("/opt/homebrew/bin/modal"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "modal"


def find_modal_script() -> Path:
    """Ищет transcribe_modal.py в корне репозитория или ~/.local/bin."""
    # Текущая директория репозитория
    here = Path(__file__).resolve().parent.parent.parent / "transcribe_modal.py"
    if here.exists():
        return here
    local = Path.home() / ".local/bin/transcribe_modal.py"
    if local.exists():
        return local
    return Path("transcribe_modal.py")


class ModalWhisperEngine(BaseTranscriber):
    """Адаптер для Whisper (large-v3, large-v3-turbo, large-v2) на Modal GPU."""

    MODELS = {
        "whisper-v3": ("large-v3", "Whisper large-v3"),
        "whisper-turbo": ("large-v3-turbo", "Whisper large-v3-turbo"),
        "whisper-v2": ("large-v2", "Whisper large-v2"),
    }

    def __init__(self, key: str = "whisper-v3"):
        if key not in self.MODELS:
            raise ValueError(f"Неизвестная модель Whisper: {key}. Доступны: {list(self.MODELS)}")
        self._key = key
        self._model_name, self._label = self.MODELS[key]

    @property
    def name(self) -> str:
        return self._key

    @property
    def label(self) -> str:
        return self._label

    def build_cmd(
        self,
        audio_path: Path,
        output_path: Path,
        language: str = "ru",
        speakers: Optional[int] = None,
    ) -> list[str]:
        modal_bin = find_modal_bin()
        script = find_modal_script()
        cmd = [
            modal_bin, "run", str(script),
            "--file", str(audio_path),
            "--language", language,
            "--model", self._model_name,
            "--output", str(output_path),
        ]
        if speakers:
            cmd += ["--speakers", str(speakers)]
        return cmd

    def transcribe(
        self,
        audio_path: Path,
        language: str = "ru",
        speakers: Optional[int] = None,
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> EngineResult:
        out_dir = output_dir or audio_path.parent
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") if not dry_run else "<dry-run>"
        out_txt = out_dir / f"{audio_path.stem}_транскрибация_{self._model_name}_{stamp}.txt"
        words_file = out_dir / f"{audio_path.stem}_транскрибация_{self._model_name}_{stamp}_слова.json"

        cmd = self.build_cmd(audio_path, out_txt, language=language, speakers=speakers)

        if dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=True,
                transcript_path=out_txt,
                words_path=words_file,
            )

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return EngineResult(
                    engine_key=self.name,
                    label=self.label,
                    success=False,
                    error=f"Modal завершился с кодом {res.returncode}:\n{res.stderr[-500:]}",
                )
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=True,
                transcript_path=out_txt,
                words_path=words_file if words_file.exists() else None,
            )
        except Exception as e:
            return EngineResult(
                engine_key=self.name,
                label=self.label,
                success=False,
                error=str(e),
            )
