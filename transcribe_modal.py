import os
import sys
import shutil
from pathlib import Path
import modal

# 1. Постоянный том для кеширования весов моделей (Whisper, wav2vec2, Pyannote, NLTK)
# Модели скачиваются 1 раз и сохраняются навсегда, повторные запуски стартуют за секунды!
model_volume = modal.Volume.from_name("whisperx-models-cache", create_if_missing=True)

# 2. Неизменяемый образ с поддержкой CUDA и стабильными версиями библиотек
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "git+https://github.com/m-bain/whisperx.git",
        "pyannote.audio==3.3.2",
    )
)

app = modal.App("whisperx-transcriber")

@app.function(
    image=image,
    gpu="A10G",  # Высокопроизводительная серверная видеокарта Nvidia A10G (24 GB)
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache": model_volume},
)
def process_audio(
    audio_bytes: bytes,
    filename: str,
    language: str = "ru",
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
):
    import tempfile
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["TORCH_HOME"] = "/root/.cache/torch"

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    temp_audio_path = None
    try:
        suffix = Path(filename).suffix or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name

        print(f"--> [1/3] Загрузка и распознавание аудио ({filename}) с моделью large-v2...")
        audio = whisperx.load_audio(temp_audio_path)
        
        # 1. Распознавание речи
        whisper_model = whisperx.load_model(
            "large-v2",
            device=device,
            compute_type=compute_type,
            language=language if language != "auto" else None,
            download_root="/root/.cache/torch/whisperx",
        )
        result = whisper_model.transcribe(audio, batch_size=16)
        detected_lang = result.get("language", language)
        print(f"--> Язык распознан: {detected_lang}")

        # 2. Выравнивание таймкодов (word alignment)
        print("--> [2/3] Выравнивание таймкодов (forced alignment)...")
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

        # 3. Диаризация (определение спикеров)
        print("--> [3/3] Определение спикеров нейросетью Pyannote...")
        try:
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=hf_token,
                device=device,
            )
            if num_speakers:
                min_spk = num_speakers
                max_spk = num_speakers
            else:
                min_spk = min_speakers
                max_spk = max_speakers

            diarize_segments = diarize_model(audio, min_speakers=min_spk, max_speakers=max_spk)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as diarize_err:
            err_msg = str(diarize_err)
            if "speaker-diarization-community-1" in err_msg or "403" in err_msg or "Gated" in err_msg:
                raise RuntimeError(
                    "❌ Ошибка доступа Hugging Face: требуется подтвердить доступ к модели.\n"
                    "Пожалуйста, откройте ссылку в браузере:\n"
                    "👉 https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                    "и нажмите кнопку «Agree and access repository» (это бесплатно).\n"
                    "После этого повторите запуск транскрибации."
                ) from diarize_err
            print(f"Внимание: Ошибка диаризации ({diarize_err}), форматируем без разделения по спикерам.")

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
):
    path = Path(file).expanduser().resolve()
    if not path.exists():
        print(f"Ошибка: Файл '{file}' не найден!")
        sys.exit(1)

    print(f"\n🚀 Отправка аудио в облачный GPU (Modal A10G): {path.name}")
    print(f"   Размер: {path.stat().st_size / (1024 * 1024):.1f} МБ")
    print(f"   Язык: {language}")
    if speakers:
        print(f"   Количество спикеров: {speakers}")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    formatted_text = process_audio.remote(
        audio_bytes=audio_bytes,
        filename=path.name,
        language=language,
        num_speakers=speakers,
    )

    out_path = Path(output) if output else path.with_name(f"{path.stem}_транскрибация.txt")
    out_path.write_text(formatted_text, encoding="utf-8")

    print("\n" + "="*50)
    print("✅ ТРАНСКРИБАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("="*50)
    print(f"📄 Результат сохранен в: {out_path}\n")
    print("--- Первые реплики диалога ---")
    preview_lines = formatted_text.splitlines()[:15]
    print("\n".join(preview_lines))
    print("...\n" + "="*50)
