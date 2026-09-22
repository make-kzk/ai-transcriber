"""Вызов модели для разбора по брифу.

Отделён от сборки брифа: бриф можно собрать, прочитать и поправить, ничего
не потратив. Здесь только отправка и приём.
"""
import os
from pathlib import Path

MODEL = "claude-opus-5"


def available() -> tuple[bool, str]:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, ("не установлен пакет anthropic — выполните install.sh "
                       "или: uv pip install --python ~/.modal-venv/bin/python anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, ("не задан ANTHROPIC_API_KEY — добавьте ключ в ~/.bash_profile "
                       "и откройте окно заново")
    return True, ""


def estimate_cost(system: str, prompt: str) -> str:
    """Грубая оценка до отправки: платить будет пользователь, он должен знать."""
    # Русский текст — примерно 2.5 символа на токен; цены Opus 5 за миллион.
    tokens_in = (len(system) + len(prompt)) / 2.5
    cost = tokens_in / 1e6 * 5 + 3000 / 1e6 * 25
    return f"~{cost:.2f} $ (примерно {tokens_in/1000:.0f} тыс. токенов на входе)"


def analyze(system: str, prompt: str, model: str = MODEL) -> str:
    import anthropic

    client = anthropic.Anthropic()
    # Стримом: на входе вся расшифровка, обычный запрос упирается в таймаут.
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError("Модель отказалась отвечать на этот материал.")
    return "\n".join(b.text for b in message.content if b.type == "text")


def save(text: str, transcript: Path) -> Path:
    out = transcript.with_name(transcript.stem + "_разбор.md")
    out.write_text(text + "\n", encoding="utf-8")
    return out
