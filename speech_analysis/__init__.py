"""Разбор речи по готовой расшифровке.

Модуль намеренно не зависит от системы транскрибации: он знает только
формат файла `[MM:SS - MM:SS] SPEAKER_XX:` и формат пословных таймкодов.
Благодаря этому его можно дорабатывать отдельно и применять к расшифровке
любого происхождения.

    from speech_analysis import analyze
    data, text = analyze(Path("расшифровка.txt"), names=["Тая", "Максим"])
"""
from pathlib import Path

from . import acoustics, config, metrics, report

__all__ = ["analyze", "acoustics", "config", "metrics", "report"]


def analyze(transcript: Path, words: Path | None = None,
            names: list[str] | None = None, cfg: dict | None = None,
            audio: Path | None = None):
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
        "inner": None,
        "acoustics": None,
        "acoustics_note": None,
    }

    word_list = metrics.parse_words(words) if words else None
    if word_list:
        data["inner"] = metrics.inside_turns(word_list, name_map, cfg)

    # Акустика требует и звука, и пословных таймкодов: без них неизвестно,
    # какой отрезок записи кому принадлежит.
    if audio and cfg["show"].get("acoustics", True):
        ok, why = acoustics.available()
        if not ok:
            data["acoustics_note"] = why
        elif not word_list:
            data["acoustics_note"] = ("нет пословных таймкодов — непонятно, "
                                      "какой участок записи чей")
        else:
            try:
                data["acoustics"] = acoustics.measure(Path(audio), word_list,
                                                      name_map, cfg)
            except Exception as e:
                data["acoustics_note"] = f"не удалось измерить: {e}"

    return data, report.render(data, cfg)
