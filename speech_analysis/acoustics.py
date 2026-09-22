"""Акустика: высота тона, громкость, темп артикуляции.

Это честная замена «энергичности» и «дикции»: вместо ярлыка от нейросети —
три измеримые величины, которые можно сверять между встречами.

Измерения делает Praat через parselmouth — эталонный инструмент фонетики,
а не самодельная обработка сигнала. Библиотека необязательная: без неё
остальной разбор работает, а этот блок сообщает, чего не хватает.
"""
import re
import shutil
import statistics as st
import subprocess
import tempfile
from pathlib import Path

# Гласные как приближение слогов: в русском их число почти совпадает
# с числом слогов. Точный слогораздел здесь не нужен — нужна скорость.
VOWELS = set("аеёиоуыэюяaeiouy")


def available() -> tuple[bool, str]:
    """Есть ли всё нужное для измерений."""
    try:
        import parselmouth  # noqa: F401
    except ImportError:
        return False, "не установлен praat-parselmouth"
    if not shutil.which("ffmpeg"):
        return False, "не найден ffmpeg — им приводим звук к пригодному виду"
    return True, ""


def _to_wav(audio: Path) -> Path:
    """Praat читает не всякий контейнер, поэтому приводим к моно 16 кГц."""
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
         "-ac", "1", "-ar", "16000", str(tmp)],
        check=True,
    )
    return tmp


def _semitones(values, base) -> float:
    """Разброс тона в полутонах: в герцах мужской и женский голос несравнимы."""
    import math
    if not values or base <= 0:
        return 0.0
    rel = [12 * math.log2(v / base) for v in values if v > 0]
    return st.pstdev(rel) if len(rel) > 1 else 0.0


def measure(audio: Path, words, names: dict, cfg: dict) -> dict:
    """Метрики по каждому говорящему. words — пословные таймкоды."""
    import numpy as np
    import parselmouth

    wav = _to_wav(Path(audio))
    try:
        snd = parselmouth.Sound(str(wav))
        pitch = snd.to_pitch(time_step=0.01)
        intensity = snd.to_intensity(time_step=0.01)

        f0 = pitch.selected_array["frequency"]           # 0 там, где нет голоса
        f0_t = pitch.xs()
        db = np.asarray(intensity.values).flatten()
        db_t = intensity.xs()

        per = {}
        for w in words:
            spk = names.get(w.get("speaker"), w.get("speaker") or "—")
            p = per.setdefault(spk, {"f0": [], "db": [], "syllables": 0, "voiced": 0.0})
            a, b = w["start"], w["end"]
            if b <= a:
                continue
            sel = (f0_t >= a) & (f0_t <= b)
            p["f0"].extend(v for v in f0[sel] if v > 0)
            sel_db = (db_t >= a) & (db_t <= b)
            # Тишину в среднюю громкость не берём: она занижает её тем сильнее,
            # чем больше в записи пауз, и делает людей несравнимыми.
            p["db"].extend(v for v in db[sel_db] if v > 25)
            p["syllables"] += sum(1 for ch in w["text"].lower() if ch in VOWELS)
            p["voiced"] += b - a

        out = {}
        for spk, p in per.items():
            median_f0 = st.median(p["f0"]) if p["f0"] else 0
            out[spk] = {
                "f0_median": median_f0,
                "f0_spread_st": _semitones(p["f0"], median_f0),
                "db_median": st.median(p["db"]) if p["db"] else 0,
                "db_spread": st.pstdev(p["db"]) if len(p["db"]) > 1 else 0,
                "articulation": p["syllables"] / p["voiced"] if p["voiced"] else 0,
                "voiced_seconds": p["voiced"],
            }
        return out
    finally:
        wav.unlink(missing_ok=True)
