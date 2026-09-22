"""Unit-тесты для парсеров адаптеров ASR (ElevenLabs, Gemini) и вспомогательных утилит."""
import unittest
from ai_transcriber.core.utils import SpeakerMap, assign_words_to_segments, fitted_transcript, mmss
from ai_transcriber.engines.elevenlabs import extract_words as el_extract_words
from ai_transcriber.engines.gemini import extract_words as gemini_extract_words, _seconds


class TestElevenLabsAdapter(unittest.TestCase):
    """Проверка парсинга ответа ElevenLabs Scribe STT."""

    def test_extract_words_clean(self):
        payload = {
            "language_code": "rus",
            "audio_duration_secs": 15.5,
            "words": [
                {"text": "Привет", "start": 0.5, "end": 1.0, "type": "word", "speaker_id": "speaker_0"},
                {"text": ",", "start": None, "end": None, "type": "spacing", "speaker_id": "speaker_0"},
                {"text": "коллеги", "start": 1.2, "end": 1.8, "type": "word", "speaker_id": "speaker_0"},
                {"text": "Добрый", "start": 2.5, "end": 3.0, "type": "word", "speaker_id": "speaker_1"},
                {"text": "день", "start": 3.1, "end": 3.6, "type": "word", "speaker_id": "speaker_1"},
            ],
        }
        words = el_extract_words(payload)
        self.assertEqual(len(words), 4)
        self.assertEqual(words[0]["text"], "Привет")
        self.assertEqual(words[0]["start"], 0.5)
        self.assertEqual(words[0]["speaker"], "speaker_0")
        self.assertEqual(words[2]["text"], "Добрый")
        self.assertEqual(words[2]["speaker"], "speaker_1")

    def test_extract_words_empty_or_malformed(self):
        self.assertEqual(el_extract_words({}), [])
        self.assertEqual(el_extract_words({"words": []}), [])
        self.assertEqual(el_extract_words({"words": [{"type": "spacing"}]}), [])


class TestGeminiAdapter(unittest.TestCase):
    """Проверка парсинга ответа Gemini 3.5 Transcribe."""

    def test_seconds_parser(self):
        self.assertEqual(_seconds(None), None)
        self.assertEqual(_seconds(2.5), 2.5)
        self.assertEqual(_seconds(10), 10.0)
        self.assertEqual(_seconds("3.75s"), 3.75)
        self.assertEqual(_seconds("0s"), 0.0)
        self.assertEqual(_seconds("invalid"), None)

    def test_extract_words_dict_structure(self):
        mock_response = {
            "steps": [
                {
                    "content": [
                        {
                            "annotations": [
                                {
                                    "type": "word_info",
                                    "text": "Стратегия",
                                    "start_offset": "1.2s",
                                    "end_offset": "1.9s",
                                    "speaker": "SPEAKER_00",
                                },
                                {
                                    "type": "other_ann",
                                    "text": "ignored",
                                },
                                {
                                    "type": "word_info",
                                    "text": "развития",
                                    "start_offset": 2.0,
                                    "end_offset": 2.6,
                                    "speaker": "SPEAKER_00",
                                },
                            ]
                        }
                    ]
                }
            ]
        }
        words = gemini_extract_words(mock_response)
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["text"], "Стратегия")
        self.assertEqual(words[0]["start"], 1.2)
        self.assertEqual(words[0]["end"], 1.9)
        self.assertEqual(words[1]["text"], "развития")
        self.assertEqual(words[1]["start"], 2.0)
        self.assertEqual(words[1]["end"], 2.6)


class TestCoreUtils(unittest.TestCase):
    """Проверка утилит разметки, таймкодов и распределения по сегментам."""

    def test_speaker_map(self):
        smap = SpeakerMap()
        self.assertEqual(smap("host"), "SPEAKER_00")
        self.assertEqual(smap("guest"), "SPEAKER_01")
        self.assertEqual(smap("host"), "SPEAKER_00")
        self.assertEqual(len(smap), 2)

    def test_mmss_formatter(self):
        self.assertEqual(mmss(0), "00:00")
        self.assertEqual(mmss(65), "01:05")
        self.assertEqual(mmss(3605), "60:05")

    def test_assign_words_to_segments(self):
        reference = [
            {"head": "[00:00 - 00:10] SPEAKER_00:", "start": 0, "end": 10},
            {"head": "[00:12 - 00:20] SPEAKER_01:", "start": 12, "end": 20},
        ]
        words = [
            {"text": "Привет", "start": 1.0, "end": 2.0},
            {"text": "мир", "start": 3.0, "end": 4.0},
            {"text": "потерянный", "start": 11.2, "end": 11.6},  # mid=11.4: > 10+1 and < 12, true orphan
            {"text": "Ответ", "start": 13.0, "end": 14.5},
        ]
        buckets, orphans = assign_words_to_segments(words, reference)
        self.assertEqual(buckets[0], ["Привет", "мир"])
        self.assertEqual(buckets[1], ["Ответ"])
        self.assertEqual(orphans, 1)

        fitted = fitted_transcript(words, reference)
        self.assertIn("[00:00 - 00:10] SPEAKER_00:\nПривет мир", fitted)
        self.assertIn("[00:12 - 00:20] SPEAKER_01:\nОтвет", fitted)


if __name__ == "__main__":
    unittest.main()
