import os
import sys
from datetime import datetime
from pathlib import Path
import modal

# 1. Постоянный том для кеширования весов моделей (Whisper, wav2vec2, Pyannote, NLTK)
# Модели скачиваются 1 раз и сохраняются навсегда, повторные запуски стартуют за секунды!
model_volume = modal.Volume.from_name("whisperx-models-cache", create_if_missing=True)

# 2. Неизменяемый образ с поддержкой CUDA и стабильными проверенными версиями библиотек
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "whisperx==3.3.2",
        "pyannote.audio==3.3.2",
        "transformers==4.48.3",
        # ctranslate2 4.5.0 — первая версия под cuDNN 9, как в torch 2.5.1 (cudnn 9.1.0.70);
        # whisperx 3.3.2 — единственный релиз 3.x, требующий ctranslate2>=4.5.0
        "ctranslate2==4.5.0",
        # matplotlib — необъявленная зависимость pyannote.audio 3.3.2:
        # импортируется на верхнем уровне в tasks/segmentation/mixins.py
        "matplotlib==3.11.2",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "NLTK_DATA": "/cache/nltk_data",
        # CTranslate2 грузит cuDNN через динамический линковщик, а torch — по абсолютным
        # путям, поэтому библиотеки из site-packages/nvidia надо явно добавить в поиск.
        # Только через .env(): LD_LIBRARY_PATH читается при старте процесса.
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib"
        ),
    })
)

app = modal.App("whisperx-transcriber")

@app.function(
    image=image,
    gpu="A10G",  # Высокопроизводительная серверная видеокарта Nvidia A10G (24 GB)
    timeout=1800,  # первый прогон включает скачивание моделей в пустой том
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_volume},
)
def process_audio(
    audio_bytes: bytes,
    filename: str,
    language: str = "ru",
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    model_name: str = "large-v3",
):
    import time
    import inspect
    import logging
    import tempfile
    import warnings

    # Три предупреждения появляются на каждом прогоне и ничего не значат:
    # чекпоинт VAD идёт из самого пакета whisperx, формат Lightning
    # обновляется только в памяти, а TF32 pyannote выключает намеренно ради
    # воспроизводимости диаризации. В красном шуме тонут настоящие ошибки.
    # Гасим адресно — остальные предупреждения по-прежнему видны.
    warnings.filterwarnings("ignore", message=r"You are using `torch\.load` with `weights_only=False`")
    warnings.filterwarnings("ignore", message=r"Lightning automatically upgraded your loaded checkpoint")
    warnings.filterwarnings("ignore", message=r"TensorFloat-32 \(TF32\) has been disabled")
    # То же сообщение Lightning печатает и через логгер, мимо warnings.
    logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.ERROR)

    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    timings = {}
    t_run = time.time()

    def mark(stage, since):
        """Замер длительности этапа — чтобы оптимизировать по факту, а не на глаз."""
        timings[stage] = time.time() - since

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    temp_audio_path = None
    try:
        suffix = Path(filename).suffix or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name

        print(f"--> [1/3] Загрузка и распознавание аудио ({filename}) моделью {model_name}...")
        audio = whisperx.load_audio(temp_audio_path)
        
        # 1. Распознавание речи
        t_load = time.time()
        whisper_model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language if language != "auto" else None,
            download_root="/cache/whisperx",
        )
        mark("Загрузка модели", t_load)

        t_asr = time.time()
        # A10G — 24 ГБ, large-v3 в fp16 занимает около 5 ГБ: запас по батчу большой.
        result = whisper_model.transcribe(audio, batch_size=24)
        mark("Распознавание", t_asr)
        detected_lang = result.get("language", language)
        print(f"--> Язык распознан: {detected_lang}")

        # 2. Выравнивание таймкодов (word alignment)
        print("--> [2/3] Выравнивание таймкодов (forced alignment)...")
        t_align = time.time()
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device=device,
                return_char_alignments=False,
            )
        except Exception as align_err:
            print(f"Внимание: Выравнивание пропущено ({align_err}), используем базовые сегменты.")
        mark("Выравнивание", t_align)

        # 3. Диаризация (определение спикеров)
        print("--> [3/3] Определение спикеров нейросетью Pyannote...")
        t_diar = time.time()
        try:
            sig = inspect.signature(DiarizationPipeline.__init__)
            auth_kw = {}
            if "token" in sig.parameters:
                auth_kw["token"] = hf_token
            elif "use_auth_token" in sig.parameters:
                auth_kw["use_auth_token"] = hf_token

            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                device=device,
                **auth_kw,
            )
            if num_speakers:
                min_spk = num_speakers
                max_spk = num_speakers
            else:
                min_spk = min_speakers
                max_spk = max_speakers

            diarize_segments = diarize_model(audio, min_speakers=min_spk, max_speakers=max_spk)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            mark("Диаризация", t_diar)
        except Exception as diarize_err:
            err_msg = str(diarize_err)
            if "403" in err_msg or "Gated" in err_msg or "gated" in err_msg or "401" in err_msg:
                raise RuntimeError(
                    "❌ Нет доступа к моделям Pyannote на Hugging Face.\n"
                    "Откройте обе страницы под аккаунтом, чей токен лежит в секрете huggingface-secret,\n"
                    "и нажмите «Agree and access repository»:\n"
                    "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  https://huggingface.co/pyannote/segmentation-3.0\n"
                    "Если доступ уже выдан — проверьте, что HF_TOKEN в секрете read-токен "
                    "с разрешением на публичные gated-репозитории.\n"
                    f"Исходная ошибка: {diarize_err}"
                ) from diarize_err
            raise RuntimeError(f"❌ Ошибка на шаге диаризации (Pyannote): {diarize_err}") from diarize_err

        # Фиксация кеша в persistent volume
        try:
            model_volume.commit()
        except Exception:
            pass

        # Форматирование читаемого диалога
        output_lines = []
        current_speaker = None
        current_text = []
        current_start = 0.0
        current_end = 0.0

        for segment in result.get("segments", []):
            spk = segment.get("speaker", "Спикер")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)

            if not text:
                continue

            if spk == current_speaker and current_speaker is not None:
                current_text.append(text)
                current_end = end
            else:
                if current_speaker is not None:
                    sm, ss = divmod(int(current_start), 60)
                    em, es = divmod(int(current_end), 60)
                    output_lines.append(f"[{sm:02d}:{ss:02d} - {em:02d}:{es:02d}] {current_speaker}:\n{' '.join(current_text)}\n")
                current_speaker = spk
                current_text = [text]
                current_start = start
                current_end = end

        if current_speaker is not None and current_text:
            sm, ss = divmod(int(current_start), 60)
            em, es = divmod(int(current_end), 60)
            output_lines.append(f"[{sm:02d}:{ss:02d} - {em:02d}:{es:02d}] {current_speaker}:\n{' '.join(current_text)}\n")

        formatted_text = "\n".join(output_lines)

        total = time.time() - t_run
        print("--> Время по этапам:")
        for stage, sec in timings.items():
            print(f"      {stage}: {sec:.0f} с ({sec / total * 100:.0f}%)")
        print(f"      Итого в контейнере: {total:.0f} с")

        return formatted_text

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


@app.local_entrypoint()
def main(
    file: str,
    language: str = "ru",
    speakers: int = None,
    output: str = None,
    model: str = "large-v3",
):
    path = Path(file).expanduser().resolve()
    if not path.exists():
        print(f"Ошибка: Файл '{file}' не найден!")
        sys.exit(1)

    print(f"\n🚀 Отправка аудио в облачный GPU (Modal A10G): {path.name}")
    print(f"   Размер: {path.stat().st_size / (1024 * 1024):.1f} МБ")
    print(f"   Язык: {language}")
    print(f"   Модель: {model}")
    if speakers:
        print(f"   Количество спикеров: {speakers}")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    formatted_text = process_audio.remote(
        audio_bytes=audio_bytes,
        filename=path.name,
        language=language,
        num_speakers=speakers,
        model_name=model,
    )

    if output:
        out_path = Path(output).expanduser()
    else:
        # Дата и время прогона в имени: повторная транскрибация того же аудио
        # больше не затирает предыдущий результат молча.
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_path = path.with_name(f"{path.stem}_транскрибация_{model}_{stamp}.txt")

    # Страховка на случай двух прогонов в одну минуту или явного --output.
    if out_path.exists():
        base, suffix = out_path.with_suffix(""), out_path.suffix
        n = 2
        while out_path.exists():
            out_path = base.with_name(f"{base.name}_{n}{suffix}")
            n += 1

    out_path.write_text(formatted_text, encoding="utf-8")

    print("\n" + "="*50)
    print("✅ ТРАНСКРИБАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("="*50)
    print(f"📄 Результат сохранен в: {out_path}\n")
    print("--- Первые реплики диалога ---")
    preview_lines = formatted_text.splitlines()[:15]
    print("\n".join(preview_lines))
    print("...\n" + "="*50)
