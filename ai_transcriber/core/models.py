"""Типизированные модели данных для ai-transcriber."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"text": self.text, "start": self.start, "end": self.end}
        if self.speaker:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Word":
        return cls(
            text=data.get("text") or data.get("word") or "",
            start=float(data["start"]),
            end=float(data["end"]),
            speaker=data.get("speaker"),
        )


@dataclass
class Segment:
    start: float  # секунды
    end: float    # секунды
    speaker: str
    text: str

    @property
    def time_label(self) -> str:
        s_m, s_s = divmod(int(self.start), 60)
        e_m, e_s = divmod(int(self.end), 60)
        return f"[{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}] {self.speaker}:"


@dataclass
class DiffItem:
    line_num: int
    time_tag: str
    speaker: str
    base_text: str
    other_text: str
    category: Literal["critical", "content", "noise"]


@dataclass
class DiffSummary:
    base_name: str
    other_name: str
    critical_count: int = 0
    content_count: int = 0
    noise_count: int = 0
    unmatched_words_other: int = 0
    items: list[DiffItem] = field(default_factory=list)


@dataclass
class EngineResult:
    engine_key: str
    label: str
    success: bool
    transcript_path: Optional[Path] = None
    words_path: Optional[Path] = None
    fitted_path: Optional[Path] = None
    error: Optional[str] = None
