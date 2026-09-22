"""Разбор по брифу через Google AI Studio (Gemini).

Отделён от сборки брифа: бриф можно собрать, прочитать и поправить, ничего
не потратив. Здесь только отправка и приём.

Gemini выбран за то, чего нет у текстовых моделей: он принимает аудио
напрямую. Разбор по записи видит интонацию, паузы и перебивания сам, без
наших метрик-посредников — а стоит аудио 32 токена в секунду, то есть
дешевле, чем кажется.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

MODELS = {
    "gemini-3.8-flash": "Gemini 3.8 Flash — быстрый, $0.75/$3.75 за млн токенов",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro — точнее и дороже",
}
DEFAULT_MODEL = "gemini-3.8-flash"

# Тарификация аудио: 32 токена в секунду записи.
AUDIO_TOKENS_PER_SEC = 32
PRICES = {  # доллары за миллион токенов: вход, выход
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-pro-preview": (4.00, 18.00),
}


def available() -> tuple[bool, str]:
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False, ("не установлен пакет google-genai — выполните install.sh "
                       "или: uv pip install --python ~/.modal-venv/bin/python google-genai")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False, ("не задан GEMINI_API_KEY — ключ выдаётся в Google AI Studio, "
                       "положите его в ~/.bash_profile и откройте окно заново")
    return True, ""


def audio_seconds(path: Path) -> float:
    """Длительность записи — нужна, чтобы посчитать цену до отправки."""
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def estimate_cost(system: str, prompt: str, audio: Path | None,
                  model: str = DEFAULT_MODEL) -> str:
    """Грубая оценка до отправки: платит пользователь, он должен знать заранее."""
    price_in, price_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    # Русский текст — примерно 2.5 символа на токен.
    tokens = (len(system) + len(prompt)) / 2.5
    parts = [f"текст {tokens/1000:.0f} тыс."]
    if audio:
        secs = audio_seconds(audio)
        a_tokens = secs * AUDIO_TOKENS_PER_SEC
        tokens += a_tokens
        parts.append(f"аудио {a_tokens/1000:.0f} тыс. ({secs/60:.0f} мин)")
    cost = tokens / 1e6 * price_in + 4000 / 1e6 * price_out
    return f"~{cost:.2f} $ ({', '.join(parts)} токенов на входе)"


def analyze(system: str, prompt: str, audio: Path | None = None,
            model: str = DEFAULT_MODEL, on_progress=None) -> str:
    from google import genai
    from google.genai import types

    def say(text):
        if on_progress:
            on_progress(text)

    client = genai.Client()
    contents = []

    if audio:
        say("Загружаю запись…")
        uploaded = client.files.upload(file=str(audio))
        # Файл становится доступен не сразу: большая запись сначала
        # обрабатывается на стороне сервиса.
        deadline = time.time() + 600
        while uploaded.state == types.FileState.PROCESSING and time.time() < deadline:
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state != types.FileState.ACTIVE:
            raise RuntimeError(f"Запись не принята сервисом: состояние {uploaded.state}")
        contents.append(uploaded)

    contents.append(prompt)
    say("Модель читает разговор…")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=16000,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Модель вернула пустой ответ — возможно, сработал фильтр.")
    return text


def save(text: str, transcript: Path) -> Path:
    out = transcript.with_name(transcript.stem + "_разбор.md")
    out.write_text(text + "\n", encoding="utf-8")
    return out
