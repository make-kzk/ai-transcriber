"""Разбор речи по готовой расшифровке.

Модуль намеренно не зависит от системы транскрибации: он знает только
формат файла `[MM:SS - MM:SS] SPEAKER_XX:` и формат пословных таймкодов.
Благодаря этому его можно дорабатывать отдельно и применять к расшифровке
любого происхождения.

    from speech_analysis import analyze
    data, text = analyze(Path("расшифровка.txt"), names=["Тая", "Максим"])
"""
from pathlib import Path

from . import config, metrics, report

__all__ = ["analyze", "config", "metrics", "report"]


def analyze(transcript: Path, words: Path | None = None,
            names: list[str] | None = None, cfg: dict | None = None):
    """Считает метрики и возвращает (данные, готовый текст отчёта)."""
    transcript = Path(transcript)
    cfg = cfg or config.load()

    segments = metrics.parse_transcript(transcript)
    if not segments:
        raise ValueError("В файле не найдено ни одной реплики — это расшифровка?")

    name_map = {f"SPEAKER_{i:02d}": n.strip()
                for i, n in enumerate(names or []) if n.strip()}

    if words is None:
        guess = transcript.with_name(transcript.stem + "_слова.json")
        words = guess if guess.exists() else None

    pauses, overlaps = metrics.transitions(segments, name_map)
    data = {
        "source": transcript.name,
        "span": segments[-1]["end"] - segments[0]["start"],
        "turns": len(segments),
        "people": metrics.per_speaker(segments, name_map, cfg),
        "pauses": pauses,
        "overlaps": overlaps,
        "inner": metrics.inside_turns(metrics.parse_words(words), name_map, cfg)
                 if words else None,
    }
    return data, report.render(data, cfg)
