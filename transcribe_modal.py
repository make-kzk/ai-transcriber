import os
import sys
import argparse
from pathlib import Path
import modal

# 1. Задаем неизменяемый Docker-образ с CUDA и стабильными версиями WhisperX и Pyannote
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "git+https://github.com/m-bain/whisperx.git",
    )
)

app = modal.App("whisperx-transcriber")

@app.function(
    image=image,
    gpu="A10G",  # Высокопроизводительная серверная видеокарта Nvidia A10G (24 GB)
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
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

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    suffix = Path(filename).suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_audio_path = temp_file.name

    try:
        print(f"--> [1/3] Загрузка и распознавание аудио ({filename}) с моделью large-v2...")
        audio = whisperx.load_audio(temp_audio_path)
        
        # 1. Распознавание речи
        whisper_model = whisperx.load_model(
            "large-v2",
            device=device,
            compute_type=compute_type,
            language=language if language != "auto" else None,
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

        # 3. Диаризация (разделение по голосам)
        print("--> [3/3] Определение спикеров нейросетью Pyannote...")
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        except Exception:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(token=hf_token, device=device)
        
        if num_speakers:
            min_spk = num_speakers
            max_spk = num_speakers
        else:
            min_spk = min_speakers
            max_spk = max_speakers

        diarize_segments = diarize_model(audio, min_speakers=min_spk, max_speakers=max_spk)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Форматирование аккуратного читаемого диалога
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
        return formatted_text, result

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


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

    formatted_text, raw_result = process_audio.remote(
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
