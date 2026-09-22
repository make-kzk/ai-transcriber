"""Настройки разбора речи.

Хранятся в домашней папке, а не в репозитории: это пользовательские
предпочтения, они не должны приезжать с обновлением кода и уезжать в git.
"""
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "ai-transcriber" / "speech_analysis.json"

DEFAULTS = {
    # Пауза внутри реплики, которую считаем заметной.
    "pause_short": 0.5,
    # Пауза, которую считаем долгой — заминка, а не дыхание.
    "pause_long": 1.5,
    # Реплики короче этого не годятся для подсчёта темпа: время округлено
    # до секунды, и на коротких отрезках округление больше самой величины.
    "min_turn_for_rate": 5.0,
    # Паразиты — только настоящие. Служебные слова «и», «а», «как», «то»
    # сюда не входят: их частота говорит о языке, а не о качестве речи.
    "parasites": [
        "ну", "вот", "типа", "короче", "блин", "значит", "слушай",
        "прям", "прямо", "походу", "реально", "просто", "получается",
    ],
    "parasite_pairs": [
        ["как", "бы"], ["то", "есть"], ["в", "общем"],
        ["это", "самое"], ["так", "сказать"], ["на", "самом"],
    ],
    "show": {
        "balance": True,      # время в эфире, число и длина реплик
        "rate": True,         # темп речи
        "parasites": True,
        "questions": True,
        "transitions": True,  # паузы и перебивания между репликами
        "inner": True,        # паузы и запинки внутри реплики
        "acoustics": True,    # тон, громкость, темп артикуляции
    },
}


def load() -> dict:
    """Настройки пользователя поверх значений по умолчанию."""
    cfg = json.loads(json.dumps(DEFAULTS))  # глубокая копия
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cfg  # испорченный файл не должен ломать разбор
        for key, value in saved.items():
            if key == "show" and isinstance(value, dict):
                cfg["show"].update(value)
            elif key in cfg:
                cfg[key] = value
    return cfg


def save(cfg: dict) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return CONFIG_PATH
