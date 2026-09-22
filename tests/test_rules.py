"""Тесты на правила разметки и разбора.

Только стандартная библиотека — запускается без установки чего-либо:

    python -m unittest discover tests

Почему именно эти правила. За разработку дважды случалось, что
правдоподобное число оказывалось артефактом: шаблон заполненных пауз
ловил обычные слова «а», «у», «на» и дал у Whisper 131 несуществующую
запинку; список паразитов включал служебные слова и дал 26% вместо 6%.
Оба раза ошибка нашлась случайно, на реальных данных. Здесь она найдётся
сразу.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_transcripts as ct
import transcript_utils as tu
from speech_analysis import config as sa_config
from speech_analysis import metrics as sm


class TestDivergenceRules(unittest.TestCase):
    """Классификация расхождений между расшифровками."""

    def check(self, left, right, expected):
        self.assertEqual(ct.classify(left.split(), right.split()), expected,
                         f"«{left}» против «{right}»")

    def test_lost_negation_is_critical(self):
        # Ради этого случая всё и затевалось: смысл переворачивается,
        # а текст остаётся гладким.
        self.check("не", "—".replace("—", ""), "critical")
        self.check("они", "не", "critical")

    def test_negation_on_both_sides_is_not_critical(self):
        # «Никакой замене» и «ни о какой замене» значат одно и то же.
        self.check("никакой", "ни о какой", "content")
        self.check("не такого", "нет такого", "content")

    def test_inflected_negative_pronouns_count_as_negation(self):
        for word in ("никакой", "никогда", "никем", "ничему", "нисколько"):
            self.assertTrue(sm.norm(word) and ct.has_negation([word]), word)

    def test_ne_znayu_is_filler_not_negation(self):
        # «не знаю» — оборот-паразит, его «не» отрицанием не считается.
        self.check("не знаю", "", "noise")
        self.check("там не знаю", "", "noise")

    def test_same_number_written_differently(self):
        self.check("2000", "2 тысячи", "noise")
        self.check("3-5", "три пять", "noise")
        self.check("20", "двадцать", "noise")

    def test_different_number_is_critical(self):
        self.check("50", "15", "critical")

    def test_pure_filler_is_noise(self):
        self.check("ну вот", "", "noise")
        self.check("", "то есть", "noise")

    def test_content_word_is_flagged(self):
        self.check("задержки", "затяжки", "content")

    def test_kolloquial_forms_are_same_word(self):
        self.assertEqual(ct.norm("нету"), ct.norm("нет"))


class TestBoundaryShift(unittest.TestCase):
    """Текст, уехавший в соседнюю реплику, — не потеря."""

    def segments(self, texts):
        return [{"text": t} for t in texts]

    def test_long_fragment_found_next_door(self):
        segs = self.segments(["", "я не оценивал этих кандидатов", ""])
        self.assertTrue(tu and ct.moved_to_neighbour(
            ["я", "не", "оценивал", "этих"], segs, 0))

    def test_short_fragment_is_never_a_boundary_shift(self):
        # Без этого порога правило съедало одиночное «не» — то самое,
        # которое мы и ищем: оно есть в любой соседней реплике.
        segs = self.segments(["там не было", "", "не знаю"])
        self.assertFalse(ct.moved_to_neighbour(["не"], segs, 1))


class TestHesitations(unittest.TestCase):
    """Запинки различаются по форме слова."""

    def test_ordinary_words_are_not_hesitations(self):
        for word in ("а", "у", "на", "ну", "но", "мы", "это", "он"):
            self.assertIsNone(sm.classify_hesitation("", word), word)

    def test_filled_pauses(self):
        for word in ("э", "ээ", "эм", "мм", "ааа", "а-а", "э-э-э"):
            self.assertEqual(sm.classify_hesitation("", word), "заполненная пауза", word)

    def test_word_fragments(self):
        for word in ("оцен--", "кон-"):
            self.assertEqual(sm.classify_hesitation("", word), "обрыв слова", word)

    def test_repeat_needs_matching_previous_word(self):
        self.assertEqual(sm.classify_hesitation("что", "что"), "повтор слова")
        self.assertIsNone(sm.classify_hesitation("и", "а"))

    def test_parasites_exclude_function_words(self):
        # Служебные слова говорят о языке, а не о качестве речи.
        words = "и а как то есть ну вот".split()
        singles = set(sa_config.DEFAULTS["parasites"])
        pairs = sa_config.DEFAULTS["parasite_pairs"]
        self.assertEqual(sm.count_parasites(words, singles, pairs), 3)  # то есть, ну, вот


class TestSpeakerMapping(unittest.TestCase):
    """Метки говорящих у провайдеров разные."""

    def test_numbered_by_first_appearance(self):
        m = tu.SpeakerMap()
        self.assertEqual(m("speaker_1"), "SPEAKER_00")
        self.assertEqual(m("speaker_0"), "SPEAKER_01")
        self.assertEqual(m("speaker_1"), "SPEAKER_00")

    def test_string_roles_work(self):
        m = tu.SpeakerMap()
        self.assertEqual(m("Агент"), "SPEAKER_00")
        self.assertEqual(m("Клиент"), "SPEAKER_01")
        self.assertEqual(len(m), 2)

    def test_zero_and_one_based_do_not_collide(self):
        # Прежний код сворачивал speaker_0 и speaker_1 в один SPEAKER_00,
        # и вся расшифровка ElevenLabs склеивалась в одну реплику.
        m = tu.SpeakerMap()
        self.assertNotEqual(m("speaker_0"), m("speaker_1"))


class TestFitting(unittest.TestCase):
    """Раскладка слов внешней системы по репликам эталона."""

    REF = [
        {"head": "[00:00 - 00:02] SPEAKER_00:", "start": 0, "end": 2},
        {"head": "[00:10 - 00:12] SPEAKER_01:", "start": 10, "end": 12},
    ]

    def test_words_land_in_matching_segment(self):
        words = [{"text": "раз", "start": 0.5, "end": 1.0},
                 {"text": "два", "start": 10.5, "end": 11.0}]
        out = tu.fitted_transcript(words, self.REF)
        self.assertIn("раз", out.split("[00:10")[0])
        self.assertIn("два", out.split("[00:10")[1])

    def test_word_outside_any_segment_is_reported(self):
        # Молча терять слова нельзя: так не видно случаев, когда вторая
        # система услышала то, чего эталон не заметил вовсе.
        words = [{"text": f"слово{i}", "start": 50 + i, "end": 50.5 + i}
                 for i in range(5)]
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tu.fitted_transcript(words, self.REF)
        self.assertIn("вне реплик эталона", buf.getvalue())


class TestConfig(unittest.TestCase):
    """Настройки разбора речи."""

    def test_defaults_survive_broken_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "broken.json"
            path.write_text("{это не json", encoding="utf-8")
            original, sa_config.CONFIG_PATH = sa_config.CONFIG_PATH, path
            try:
                cfg = sa_config.load()
            finally:
                sa_config.CONFIG_PATH = original
            self.assertEqual(cfg["pause_short"], sa_config.DEFAULTS["pause_short"])

    def test_user_values_override_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cfg.json"
            path.write_text(json.dumps({"pause_short": 0.9, "show": {"rate": False}}),
                            encoding="utf-8")
            original, sa_config.CONFIG_PATH = sa_config.CONFIG_PATH, path
            try:
                cfg = sa_config.load()
            finally:
                sa_config.CONFIG_PATH = original
            self.assertEqual(cfg["pause_short"], 0.9)
            self.assertFalse(cfg["show"]["rate"])
            self.assertTrue(cfg["show"]["balance"])   # остальное не затёрто


class TestTranscriptParsing(unittest.TestCase):
    """Формат расшифровки — контракт между модулями."""

    SAMPLE = ("[00:00 - 00:03] SPEAKER_00:\nПривет.\n\n"
              "[00:05 - 00:09] SPEAKER_01:\nДобрый день.\nКак дела?\n")

    def test_parses_time_speaker_and_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.txt"
            path.write_text(self.SAMPLE, encoding="utf-8")
            segs = sm.parse_transcript(path)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[1]["start"], 5)
        self.assertEqual(segs[1]["speaker"], "SPEAKER_01")
        self.assertEqual(segs[1]["text"], "Добрый день. Как дела?")

    def test_module_parses_independently_of_transcription_code(self):
        # speech_analysis не должен зависеть от системы транскрибации.
        import speech_analysis.metrics as m
        source = Path(m.__file__).read_text(encoding="utf-8")
        for forbidden in ("import transcript_utils", "import compare_transcripts",
                          "import transcribe_"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
