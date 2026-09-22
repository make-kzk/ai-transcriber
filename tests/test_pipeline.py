"""Unit-тесты для гибкого оркестратора пайплайна транскрибации."""
import json
import tempfile
import unittest
from pathlib import Path

from ai_transcriber.core.pipeline import run_pipeline, fit_words_to_reference


class TestPipelineOrchestrator(unittest.TestCase):
    """Тестирование логики оркестрации пайплайна."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        # Создаем фиктивный аудио-файл
        self.dummy_audio = self.tmp_path / "meeting_recording.m4a"
        self.dummy_audio.write_bytes(b"dummy audio data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_dry_run_multi_system(self):
        systems = ["whisper-v3", "elevenlabs", "gemini"]
        res = run_pipeline(
            audio=self.dummy_audio,
            systems=systems,
            stagger=0.0,
            dry_run=True,
            output_dir=self.tmp_path,
        )
        self.assertEqual(res["chosen"], systems)
        self.assertEqual(res["reference_system"], "whisper-v3")
        self.assertEqual(len(res["produced"]), 3)
        self.assertEqual(res["failed"], [])

    def test_pipeline_custom_base_selection(self):
        systems = ["whisper-v3", "elevenlabs"]
        res = run_pipeline(
            audio=self.dummy_audio,
            systems=systems,
            base_system="elevenlabs",
            stagger=0.0,
            dry_run=True,
            output_dir=self.tmp_path,
        )
        self.assertEqual(res["reference_system"], "elevenlabs")

    def test_fit_words_to_reference_integration(self):
        ref_file = self.tmp_path / "reference.txt"
        ref_file.write_text(
            "[00:00 - 00:05] SPEAKER_00:\nИсходный текст реплики\n\n"
            "[00:06 - 00:10] SPEAKER_01:\nВторая реплика\n",
            encoding="utf-8",
        )

        words_data = [
            {"text": "Новый", "start": 0.5, "end": 1.5},
            {"text": "вариант", "start": 2.0, "end": 3.0},
            {"text": "ответа", "start": 7.0, "end": 8.0},
        ]
        words_file = self.tmp_path / "other_words.json"
        words_file.write_text(json.dumps(words_data), encoding="utf-8")

        out_fitted = self.tmp_path / "other_fitted.txt"
        result = fit_words_to_reference(words_file, ref_file, out_fitted)
        self.assertIsNotNone(result)
        self.assertTrue(out_fitted.exists())

        content = out_fitted.read_text(encoding="utf-8")
        self.assertIn("[00:00 - 00:05] SPEAKER_00:\nНовый вариант", content)
        self.assertIn("[00:06 - 00:10] SPEAKER_01:\nответа", content)


if __name__ == "__main__":
    unittest.main()
