import os
import sys
import shlex
import shutil
import tempfile
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
import speech_analysis
from speech_analysis import brief as sa_brief
from speech_analysis import config as sa_config
from speech_analysis import llm as sa_llm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# Автоматическое динамическое определение путей
HOME = Path.home()

def find_modal_bin():
    if shutil.which("modal"):
        return shutil.which("modal")
    for candidate in [
        HOME / ".local/bin/modal",
        HOME / ".modal-venv/bin/modal",
        Path("/usr/local/bin/modal"),
        Path("/opt/homebrew/bin/modal"),
    ]:
        if candidate.exists():
            return str(candidate)
    return "modal"

MODAL_BIN = find_modal_bin()

# Окно само работает в этом интерпретаторе — им же запускаем вспомогательные скрипты.
VENV_PYTHON = sys.executable

# Modal хранит логи прогонов всего сутки, поэтому складываем вывод к себе:
# разбирать упавший прогон назавтра иначе будет не по чему.
LOG_DIR = HOME / "ai-transcriber-logs"

def find_script_path():
    repo_script = Path(__file__).resolve().parent / "transcribe_modal.py"
    if repo_script.exists():
        return str(repo_script)
    local_script = HOME / ".local/bin/transcribe_modal.py"
    if local_script.exists():
        return str(local_script)
    return "transcribe_modal.py"

SCRIPT_PATH = find_script_path()


class TranscribeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Транскрибация (WhisperX + GPU)")
        self.root.geometry("700x670")
        self.root.minsize(620, 560)

        self.selected_file = None

        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        self.root.configure(bg="#F5F5F7")
        style.configure("TFrame", background="#F5F5F7")
        style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        style.configure("TLabel", background="#F5F5F7", font=("SF Pro Text", 12))
        style.configure("Card.TLabel", background="#FFFFFF", font=("SF Pro Text", 12))
        style.configure("Card.TCheckbutton", background="#FFFFFF", font=("SF Pro Text", 12))
        style.configure("Header.TLabel", background="#F5F5F7", font=("SF Pro Display", 18, "bold"), foreground="#1D1D1F")
        style.configure("SubHeader.TLabel", background="#F5F5F7", font=("SF Pro Text", 11), foreground="#86868B")
        style.configure("Status.TLabel", background="#FFFFFF", font=("SF Pro Text", 12, "bold"), foreground="#0071E3")
        style.configure("Timer.TLabel", background="#FFFFFF", font=("Menlo", 12, "bold"), foreground="#515154")

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab_transcribe = ttk.Frame(notebook)
        tab_analysis = ttk.Frame(notebook)
        notebook.add(tab_transcribe, text="  Транскрибация  ")
        notebook.add(tab_analysis, text="  Анализ речи  ")

        self._build_transcribe_tab(tab_transcribe)
        self._build_analysis_tab(tab_analysis)

    def _build_transcribe_tab(self, parent):
        main_container = ttk.Frame(parent, padding=20)
        main_container.pack(fill="both", expand=True)

        # 1. Заголовок
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 12))
        
        lbl_title = ttk.Label(header_frame, text="🎙️ Облачная AI Транскрибация", style="Header.TLabel")
        lbl_title.pack(anchor="w")
        lbl_sub = ttk.Label(header_frame, text="WhisperX + Pyannote на серверной видеокарте Nvidia A10G", style="SubHeader.TLabel")
        lbl_sub.pack(anchor="w", pady=(2, 0))

        # 2. Карточка выбора файла
        file_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        file_card.pack(fill="x", pady=(0, 10))

        lbl_file_title = ttk.Label(file_card, text="Аудио или видеофайл:", style="Card.TLabel", font=("SF Pro Text", 12, "bold"))
        lbl_file_title.pack(anchor="w", pady=(0, 6))

        file_select_box = ttk.Frame(file_card, style="Card.TFrame")
        file_select_box.pack(fill="x")

        self.btn_select = ttk.Button(file_select_box, text="Выбрать файл...", command=self.choose_file)
        self.btn_select.pack(side="left", padx=(0, 10))

        self.lbl_file_name = ttk.Label(file_select_box, text="Файл не выбран", style="Card.TLabel", foreground="#86868B")
        self.lbl_file_name.pack(side="left", fill="x", expand=True)

        # 3. Карточка параметров
        opts_card = ttk.Frame(main_container, style="Card.TFrame", padding=12)
        opts_card.pack(fill="x", pady=(0, 12))

        opts_grid = ttk.Frame(opts_card, style="Card.TFrame")
        opts_grid.pack(fill="x")

        ttk.Label(opts_grid, text="Спикеры:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.spk_var = tk.StringVar(value="2")
        self.spk_combo = ttk.Combobox(opts_grid, textvariable=self.spk_var, values=["Автоопределение", "2", "3", "4", "5", "6"], width=15, state="readonly")
        self.spk_combo.grid(row=0, column=1, sticky="w", padx=(0, 25), pady=3)

        ttk.Label(opts_grid, text="Язык речи:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=3)
        self.lang_var = tk.StringVar(value="Русский (ru)")
        self.lang_combo = ttk.Combobox(opts_grid, textvariable=self.lang_var, values=["Русский (ru)", "Английский (en)", "Автоопределение"], width=15, state="readonly")
        self.lang_combo.grid(row=0, column=3, sticky="w", pady=3)


        # 4. Карточка выбора систем
        sys_card = ttk.Frame(main_container, style="Card.TFrame", padding=12)
        sys_card.pack(fill="x", pady=(0, 12))

        ttk.Label(sys_card, text="Распознать системами:", style="Card.TLabel",
                  font=("SF Pro Text", 12, "bold")).pack(anchor="w", pady=(0, 6))

        sys_grid = ttk.Frame(sys_card, style="Card.TFrame")
        sys_grid.pack(fill="x")

        # Порядок тот же, что в run_pipeline.py: первая отмеченная становится
        # основой сводного документа, остальные идут в сверку.
        self.system_vars = {}
        for i, (key, label, note, default) in enumerate([
            ("elevenlabs", "ElevenLabs Scribe", "точнее на терминах, 13¢", True),
            ("whisper-v3", "Whisper large-v3", "в пределах лимита Modal", True),
            ("whisper-turbo", "Whisper large-v3-turbo", "быстрее, менее точен", False),
            ("whisper-v2", "Whisper large-v2", "прежняя версия", False),
        ]):
            var = tk.BooleanVar(value=default)
            self.system_vars[key] = var
            ttk.Checkbutton(sys_grid, text=label, variable=var,
                            style="Card.TCheckbutton").grid(
                row=i, column=0, sticky="w", pady=2)
            ttk.Label(sys_grid, text=note, style="Card.TLabel",
                      foreground="#86868B", font=("SF Pro Text", 10)).grid(
                row=i, column=1, sticky="w", padx=(14, 0))

        ttk.Label(
            sys_card,
            text="Отмеченные системы распознают запись независимо. Если их больше одной,\n"
                 "тексты сводятся и места расхождений помечаются — первая по списку идёт в основу.",
            style="Card.TLabel", foreground="#86868B", font=("SF Pro Text", 10),
        ).pack(anchor="w", pady=(8, 0))

        # 5. Панель кнопок: СТАРТ
        btn_box = ttk.Frame(main_container)
        btn_box.pack(fill="x", pady=(0, 12))

        self.btn_start = tk.Button(
            btn_box,
            text="🚀  Начать транскрибацию",
            font=("SF Pro Text", 13, "bold"),
            bg="#0071E3",
            fg="#FFFFFF",
            activebackground="#0077ED",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=15,
            pady=8,
            cursor="pointinghand",
            command=self.start_transcription,
            state="disabled",
        )
        self.btn_start.pack(fill="x", expand=True)


        # 5. Карточка статуса и прогресса
        status_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        status_card.pack(fill="x", pady=(0, 10))

        status_header = ttk.Frame(status_card, style="Card.TFrame")
        status_header.pack(fill="x", pady=(0, 6))

        self.lbl_status = ttk.Label(status_header, text="Ожидание файла", style="Status.TLabel")
        self.lbl_status.pack(side="left")


        info_box = ttk.Frame(status_card, style="Card.TFrame")
        info_box.pack(fill="x")

        self.lbl_modal_link = ttk.Label(
            info_box,
            text="🌐 Открыть панель Modal в браузере",
            style="Card.TLabel",
            foreground="#0071E3",
            cursor="pointinghand",
            font=("SF Pro Text", 11, "underline")
        )
        self.lbl_modal_link.pack(side="left")
        self.lbl_modal_link.bind("<Button-1>", lambda e: webbrowser.open("https://modal.com/apps"))

        # 6. Консоль журнала (Terminal-like Log View)
        log_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        log_card.pack(fill="both", expand=True)

        lbl_log = ttk.Label(log_card, text="Журнал работы в реальном времени:", style="Card.TLabel", font=("SF Pro Text", 11, "bold"))
        lbl_log.pack(anchor="w", pady=(0, 4))

        self.log_text = tk.Text(
            log_card,
            wrap="word",
            font=("Menlo", 11),
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#FFFFFF",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "Выберите аудиофайл и нажмите «Начать транскрибацию»...\n")
        self.log_text.config(state="disabled")

        # 7. Кнопки действий после завершения
        self.action_frame = ttk.Frame(main_container)
        self.action_frame.pack(fill="x", pady=(10, 0))

        self.btn_open_file = ttk.Button(self.action_frame, text="📄 Открыть результат", command=self.open_result_file, state="disabled")
        self.btn_open_file.pack(side="left", padx=(0, 10))

        self.btn_open_dir = ttk.Button(self.action_frame, text="📂 Показать в Finder", command=self.open_result_dir, state="disabled")
        self.btn_open_dir.pack(side="left")

    # ─────────────────────────── Анализ речи ───────────────────────────
    # Модуль speech_analysis самостоятелен: он знает только формат файла
    # расшифровки и дорабатывается отдельно от системы транскрибации.

    def _build_analysis_tab(self, parent):
        box = ttk.Frame(parent, padding=20)
        box.pack(fill="both", expand=True)

        ttk.Label(box, text="⚙️ Настройки разбора речи", style="Header.TLabel").pack(anchor="w")
        ttk.Label(box, style="SubHeader.TLabel",
                  text="Считаются только измеримые величины: темп, паузы, запинки,\n"
                       "баланс времени. Оценок вроде «уверенно» здесь нет намеренно.").pack(
            anchor="w", pady=(2, 12))

        cfg = sa_config.load()
        self.an_vars = {}

        nums = ttk.Frame(box, style="Card.TFrame", padding=12)
        nums.pack(fill="x", pady=(0, 10))
        for i, (key, label, hint) in enumerate([
            ("pause_short", "Заметная пауза, с", "короче — не считаем за паузу"),
            ("pause_long", "Долгая пауза, с", "выделяется отдельно как заминка"),
            ("min_turn_for_rate", "Мин. реплика для темпа, с",
             "на коротких округление времени искажает темп"),
        ]):
            ttk.Label(nums, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(cfg[key]))
            self.an_vars[key] = var
            ttk.Entry(nums, textvariable=var, width=8).grid(row=i, column=1, padx=(12, 12))
            ttk.Label(nums, text=hint, style="Card.TLabel", foreground="#86868B",
                      font=("SF Pro Text", 10)).grid(row=i, column=2, sticky="w")

        blocks = ttk.Frame(box, style="Card.TFrame", padding=12)
        blocks.pack(fill="x", pady=(0, 10))
        ttk.Label(blocks, text="Показывать в отчёте:", style="Card.TLabel",
                  font=("SF Pro Text", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.an_show = {}
        for i, (key, label) in enumerate([
            ("balance", "баланс времени"), ("rate", "темп речи"),
            ("parasites", "слова-паразиты"), ("questions", "вопросы"),
            ("transitions", "паузы между репликами"), ("inner", "паузы и запинки внутри"),
            ("acoustics", "акустика: тон, громкость"),
        ]):
            var = tk.BooleanVar(value=cfg["show"].get(key, True))
            self.an_show[key] = var
            ttk.Checkbutton(blocks, text=label, variable=var, style="Card.TCheckbutton").grid(
                row=1 + i // 3, column=i % 3, sticky="w", padx=(0, 18), pady=2)

        words_box = ttk.Frame(box, style="Card.TFrame", padding=12)
        words_box.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Label(words_box, text="Слова-паразиты — по одному в строке:",
                  style="Card.TLabel", font=("SF Pro Text", 12, "bold")).pack(anchor="w")
        ttk.Label(words_box, style="Card.TLabel", foreground="#86868B",
                  font=("SF Pro Text", 10),
                  text="Служебные слова «и», «а», «как», «то» сюда добавлять не стоит:\n"
                       "их частота говорит о языке, а не о качестве речи.").pack(anchor="w", pady=(0, 6))
        self.an_parasites = tk.Text(words_box, height=6, font=("Menlo", 11),
                                    relief="solid", borderwidth=1)
        self.an_parasites.pack(fill="both", expand=True)
        self.an_parasites.insert("1.0", "\n".join(cfg["parasites"]))

        audio_box = ttk.Frame(box, style="Card.TFrame", padding=12)
        audio_box.pack(fill="x", pady=(0, 10))
        ttk.Label(audio_box, text="Аудиофайл для акустики:", style="Card.TLabel",
                  font=("SF Pro Text", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(audio_box, style="Card.TLabel", foreground="#86868B",
                  font=("SF Pro Text", 10),
                  text="Нужен для измерения тона и громкости. Без него остальные метрики\n"
                       "считаются как обычно. Нужны также пословные таймкоды рядом с расшифровкой.").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.an_audio = None
        self.an_audio_lbl = ttk.Label(audio_box, text="не выбран", style="Card.TLabel",
                                      foreground="#86868B")
        ttk.Button(audio_box, text="Выбрать...", command=self._choose_analysis_audio).grid(
            row=2, column=0, sticky="w")
        self.an_audio_lbl.grid(row=2, column=1, sticky="w", padx=(10, 0))
        ttk.Button(audio_box, text="Убрать", command=self._clear_analysis_audio).grid(
            row=2, column=2, sticky="w", padx=(10, 0))

        brief_box = ttk.Frame(box, style="Card.TFrame", padding=12)
        brief_box.pack(fill="x", pady=(0, 10))
        ttk.Label(brief_box, text="Бриф: что разобрать", style="Card.TLabel",
                  font=("SF Pro Text", 12, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(brief_box, style="Card.TLabel", foreground="#86868B",
                  font=("SF Pro Text", 10),
                  text="Метрики выше считаются бесплатно и всегда. Бриф — отдельный шаг:\n"
                       "по нему модель пишет разбор для HR и нанимающего менеджера.").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(brief_box, text="Сценарий:", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        self.br_preset = tk.StringVar(value=sa_brief.PRESETS["interview"]["label"])
        preset_box = ttk.Combobox(
            brief_box, textvariable=self.br_preset, state="readonly", width=32,
            values=[p["label"] for p in sa_brief.PRESETS.values()])
        preset_box.grid(row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=3)
        preset_box.bind("<<ComboboxSelected>>", self._apply_preset)

        ttk.Label(brief_box, text="Предмет:", style="Card.TLabel").grid(row=3, column=0, sticky="w")
        self.br_subject = tk.StringVar()
        ttk.Entry(brief_box, textvariable=self.br_subject, width=40).grid(
            row=3, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=3)

        ttk.Label(brief_box, text="Кто есть кто:", style="Card.TLabel").grid(row=4, column=0, sticky="w")
        self.br_role0 = tk.StringVar()
        self.br_role1 = tk.StringVar()
        ttk.Entry(brief_box, textvariable=self.br_role0, width=18).grid(row=4, column=1, sticky="w", padx=(8, 4))
        ttk.Entry(brief_box, textvariable=self.br_role1, width=18).grid(row=4, column=2, sticky="w")
        ttk.Label(brief_box, text="первый и второй голос", style="Card.TLabel",
                  foreground="#86868B", font=("SF Pro Text", 10)).grid(row=4, column=3, sticky="w", padx=(8, 0))

        focus_frame = ttk.Frame(brief_box, style="Card.TFrame")
        focus_frame.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(focus_frame, text="Разобрать:", style="Card.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w")
        self.br_focus = {}
        for i, (key, (label, _)) in enumerate(sa_brief.FOCUS_OPTIONS.items()):
            var = tk.BooleanVar(value=False)
            self.br_focus[key] = var
            ttk.Checkbutton(focus_frame, text=label, variable=var,
                            style="Card.TCheckbutton").grid(
                row=1 + i // 2, column=i % 2, sticky="w", padx=(0, 20))

        ttk.Label(brief_box, text="Дополнительно:", style="Card.TLabel").grid(
            row=6, column=0, sticky="nw", pady=(8, 0))
        self.br_extra = tk.Text(brief_box, height=3, font=("SF Pro Text", 11),
                                relief="solid", borderwidth=1)
        self.br_extra.grid(row=6, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=(8, 0))
        brief_box.columnconfigure(3, weight=1)

        model_row = ttk.Frame(brief_box, style="Card.TFrame")
        model_row.grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.br_audio = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_row, text="Отправить вместе с записью",
                        variable=self.br_audio, style="Card.TCheckbutton").pack(side="left")
        ttk.Label(model_row, text="   Модель:", style="Card.TLabel").pack(side="left")
        self.br_model = tk.StringVar(value=sa_llm.MODELS[sa_llm.DEFAULT_MODEL])
        ttk.Combobox(model_row, textvariable=self.br_model, state="readonly", width=44,
                     values=list(sa_llm.MODELS.values())).pack(side="left", padx=(6, 0))

        ttk.Label(brief_box, style="Card.TLabel", foreground="#86868B",
                  font=("SF Pro Text", 10),
                  text="С записью модель слышит интонацию, паузы и перебивания сама —\n"
                       "аудио тарифицируется как 32 токена в секунду, это недорого.").grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(4, 0))

        brief_btns = ttk.Frame(brief_box, style="Card.TFrame")
        brief_btns.grid(row=9, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(brief_btns, text="Показать бриф", command=self.preview_brief).pack(side="left")
        ttk.Button(brief_btns, text="🧠 Разобрать по брифу...",
                   command=self.run_brief).pack(side="left", padx=8)

        self._apply_preset()

        btns = ttk.Frame(box)
        btns.pack(fill="x")
        ttk.Button(btns, text="Сохранить настройки", command=self._save_analysis_config).pack(side="left")
        ttk.Button(btns, text="Сбросить", command=self._reset_analysis_config).pack(side="left", padx=8)
        ttk.Button(btns, text="📊 Разобрать расшифровку...",
                   command=self.run_analysis).pack(side="right")

        self.an_status = ttk.Label(box, text=f"Настройки: {sa_config.CONFIG_PATH}",
                                   style="SubHeader.TLabel")
        self.an_status.pack(anchor="w", pady=(10, 0))

    def _model_key(self):
        for key, label in sa_llm.MODELS.items():
            if label == self.br_model.get():
                return key
        return sa_llm.DEFAULT_MODEL

    def _brief_audio(self):
        """Запись уходит в модель, только если она выбрана и галочка стоит."""
        return self.an_audio if (self.br_audio.get() and self.an_audio) else None

    def _preset_key(self):
        for key, p in sa_brief.PRESETS.items():
            if p["label"] == self.br_preset.get():
                return key
        return "custom"

    def _apply_preset(self, _event=None):
        """Значения по умолчанию из сценария — их можно переписать вручную."""
        preset = sa_brief.PRESETS[self._preset_key()]
        self.br_role0.set(preset["roles"][0])
        self.br_role1.set(preset["roles"][1])
        for key, var in self.br_focus.items():
            var.set(key in preset["focus"])

    def _gather_brief(self, transcript: Path):
        cfg = self._collect_analysis_config()
        # Метрики считаем без подстановки имён: в расшифровке, которая уйдёт
        # в модель, стоят SPEAKER_00/01, и обозначения должны совпадать.
        # Кто есть кто — сказано отдельной строкой в брифе.
        _, metrics_text = speech_analysis.analyze(
            transcript, cfg=cfg, audio=self.an_audio)
        roles = {"SPEAKER_00": self.br_role0.get(),
                 "SPEAKER_01": self.br_role1.get()}
        focus = [k for k, v in self.br_focus.items() if v.get()]
        return sa_brief.build(
            self._preset_key(), roles, self.br_subject.get(), focus,
            self.br_extra.get("1.0", "end"), metrics_text,
            transcript.read_text(encoding="utf-8"))

    def preview_brief(self):
        """Показывает, что именно уйдёт в модель. Бесплатно."""
        path = filedialog.askopenfilename(title="Расшифровка для брифа",
                                          filetypes=[("Расшифровка", "*.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            system, prompt = self._gather_brief(Path(path))
        except Exception as e:
            messagebox.showerror("Не удалось собрать бриф", str(e), parent=self.root)
            return
        ok, why = sa_llm.available()
        audio = self._brief_audio()
        cost = sa_llm.estimate_cost(system, prompt, audio, self._model_key())
        head = (f"Оценка стоимости: {cost}\n"
                + (f"Вместе с записью: {audio.name}\n" if audio
                   else "Без записи — только текст.\n")
                + ("" if ok else f"Разбор недоступен: {why}\n") + "\n")
        self._show_text(f"Бриф — {Path(path).name}", head + system + "\n\n" + prompt)

    def run_brief(self):
        ok, why = sa_llm.available()
        if not ok:
            messagebox.showerror("Разбор недоступен", why, parent=self.root)
            return
        path = filedialog.askopenfilename(title="Расшифровка для разбора",
                                          filetypes=[("Расшифровка", "*.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            system, prompt = self._gather_brief(Path(path))
        except Exception as e:
            messagebox.showerror("Не удалось собрать бриф", str(e), parent=self.root)
            return

        audio = self._brief_audio()
        model = self._model_key()
        what = "Расшифровка и запись уйдут" if audio else "Расшифровка уйдёт"
        if not messagebox.askokcancel(
                "Отправить в модель?",
                f"Будет израсходовано {sa_llm.estimate_cost(system, prompt, audio, model)}.\n"
                f"{what} в Google AI Studio.", parent=self.root):
            return

        def progress(text):
            self.an_status.config(text=text)
            self.root.update_idletasks()

        progress("Отправляю…")
        try:
            text = sa_llm.analyze(system, prompt, audio, model, on_progress=progress)
        except Exception as e:
            self.an_status.config(text="Разбор не получился")
            messagebox.showerror("Ошибка разбора", str(e), parent=self.root)
            return
        out = sa_llm.save(text, Path(path))
        self.an_status.config(text=f"Разбор: {out.name}")
        self._show_text(f"Разбор — {Path(path).name}", text)

    def _show_text(self, title: str, text: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("760x620")
        area = tk.Text(win, font=("SF Pro Text", 12), wrap="word", padx=10, pady=10)
        area.pack(fill="both", expand=True, padx=10, pady=10)
        area.insert("1.0", text)
        area.config(state="disabled")

    def _choose_analysis_audio(self):
        path = filedialog.askopenfilename(
            title="Аудиофайл записи",
            filetypes=[("Аудио и видео", "*.m4a *.mp3 *.wav *.mp4 *.mov *.aac *.ogg *.flac"),
                       ("Все файлы", "*.*")])
        if path:
            self.an_audio = Path(path)
            self.an_audio_lbl.config(text=self.an_audio.name, foreground="#1D1D1F")

    def _clear_analysis_audio(self):
        self.an_audio = None
        self.an_audio_lbl.config(text="не выбран", foreground="#86868B")

    def _collect_analysis_config(self):
        cfg = sa_config.load()
        for key, var in self.an_vars.items():
            try:
                cfg[key] = float(var.get().replace(",", "."))
            except ValueError:
                raise ValueError(f"«{var.get()}» — не число")
        cfg["show"] = {k: v.get() for k, v in self.an_show.items()}
        cfg["parasites"] = [w.strip().lower() for w in
                            self.an_parasites.get("1.0", "end").splitlines() if w.strip()]
        return cfg

    def _save_analysis_config(self):
        try:
            path = sa_config.save(self._collect_analysis_config())
        except ValueError as e:
            messagebox.showerror("Не сохранено", str(e), parent=self.root)
            return
        self.an_status.config(text=f"Сохранено: {path}")

    def _reset_analysis_config(self):
        if not messagebox.askokcancel("Сбросить настройки",
                                      "Вернуть значения по умолчанию?", parent=self.root):
            return
        if sa_config.CONFIG_PATH.exists():
            sa_config.CONFIG_PATH.unlink()
        defaults = sa_config.load()
        for key, var in self.an_vars.items():
            var.set(str(defaults[key]))
        for key, var in self.an_show.items():
            var.set(defaults["show"].get(key, True))
        self.an_parasites.delete("1.0", "end")
        self.an_parasites.insert("1.0", "\n".join(defaults["parasites"]))
        self.an_status.config(text="Восстановлены значения по умолчанию")

    def run_analysis(self):
        path = filedialog.askopenfilename(
            title="Выберите расшифровку",
            filetypes=[("Расшифровка", "*.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            cfg = self._collect_analysis_config()
            self.an_status.config(text="Считаю… акустика занимает до полуминуты")
            self.root.update_idletasks()
            _, text = speech_analysis.analyze(Path(path), cfg=cfg, audio=self.an_audio)
        except Exception as e:
            messagebox.showerror("Не удалось разобрать", str(e), parent=self.root)
            return

        out = Path(path).with_name(Path(path).stem + "_метрики.txt")
        out.write_text(text + "\n", encoding="utf-8")
        self.an_status.config(text=f"Отчёт: {out.name}")

        win = tk.Toplevel(self.root)
        win.title(f"Метрики: {Path(path).name}")
        win.geometry("720x600")
        area = tk.Text(win, font=("Menlo", 11), wrap="none")
        area.pack(fill="both", expand=True, padx=10, pady=10)
        area.insert("1.0", text)
        area.config(state="disabled")

    def choose_file(self):
        filetypes = [
            ("Аудио и видео", "*.m4a *.mp3 *.wav *.mp4 *.mov *.aac *.ogg *.flac *.webm"),
            ("Все файлы", "*.*")
        ]
        chosen = filedialog.askopenfilename(title="Выберите аудиофайл", filetypes=filetypes)
        if chosen:
            self.selected_file = Path(chosen)
            try:
                size_mb = self.selected_file.stat().st_size / (1024 * 1024)
                self.lbl_file_name.config(
                    text=f"{self.selected_file.name} ({size_mb:.1f} МБ)",
                    foreground="#1D1D1F"
                )
            except Exception:
                self.lbl_file_name.config(text=f"{self.selected_file.name}", foreground="#1D1D1F")

            self.btn_start.config(state="normal", bg="#0071E3")
            self.lbl_status.config(text="Готов к запуску", foreground="#1D1D1F")
            self._log_msg(f"Выбран файл: {self.selected_file.name}")

    def _log_msg(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

        # Обновление текста в статус-строке на основе сообщений
        text_lower = text.lower()
        if "[1/3]" in text or "распознавание" in text_lower:
            self.lbl_status.config(text="🎙️ [1/3] Распознавание речи WhisperX...", foreground="#0071E3")
        elif "[2/3]" in text or "выравнивание" in text_lower:
            self.lbl_status.config(text="⏱️ [2/3] Выравнивание таймкодов...", foreground="#5856D6")
        elif "[3/3]" in text or "спикеров" in text_lower or "pyannote" in text_lower:
            self.lbl_status.config(text="👥 [3/3] Определение голосов (Pyannote)...", foreground="#AF52DE")
        elif "initialized" in text_lower or "building" in text_lower or "step" in text_lower:
            self.lbl_status.config(text="⚡️ Инициализация облачного GPU...", foreground="#FF9500")

    def start_transcription(self):
        """Запускает распознавание в Терминале.

        modal отдаёт вывод пачками, когда пишет не в TTY, поэтому встроенный
        журнал не показывал ход работы. В настоящем терминале вывод живой —
        окно берёт на себя только выбор файла и систем.
        """
        if not self.selected_file:
            return

        systems = [k for k, v in self.system_vars.items() if v.get()]
        if not systems:
            messagebox.showwarning("Не выбрана система",
                                   "Отметьте хотя бы одну систему распознавания.",
                                   parent=self.root)
            return

        spk_val = self.spk_var.get()
        lang_val = self.lang_var.get()
        if "ru" in lang_val:
            lang_code = "ru"
        elif "en" in lang_val:
            lang_code = "en"
        else:
            lang_code = "auto"

        cmd = [VENV_PYTHON, str(Path(SCRIPT_PATH).resolve().parent / "run_pipeline.py"),
               str(self.selected_file), "--language", lang_code, "--systems"] + systems
        if spk_val.isdigit():
            cmd += ["--speakers", spk_val]

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = LOG_DIR / f"{self.selected_file.stem}_{stamp}.log"

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            launcher = self._write_terminal_script(cmd, log_path)
            subprocess.run(["open", "-a", "Terminal", str(launcher)], check=True)
        except Exception as e:
            self.lbl_status.config(text="❌ Не удалось открыть Терминал", foreground="#FF3B30")
            self._log_msg(f"Ошибка запуска Терминала: {e}")
            messagebox.showerror("Ошибка", f"Не удалось открыть Терминал:\n{e}", parent=self.root)
            return

        self.lbl_status.config(text="▶️ Запущено в Терминале", foreground="#34C759")
        self._log_msg("\n" + "=" * 50)
        self._log_msg(f"Запущено в Терминале: {self.selected_file.name}")
        self._log_msg(f"Язык: {lang_code}, спикеры: {spk_val}")
        self._log_msg(f"Систем выбрано: {len(systems)} — {', '.join(systems)}")
        if len(systems) > 1:
            self._log_msg("Тексты будут сведены, расхождения помечены.")
        self._log_msg("Ход работы смотрите в открывшемся окне Терминала.")
        self._log_msg(f"Журнал прогона: {log_path}")
        self._log_msg("=" * 50)

        self.btn_open_file.config(state="normal")
        self.btn_open_dir.config(state="normal")

    def _write_terminal_script(self, cmd, log_path=None):
        """Готовит .command-файл — так Терминал открывается без доступа к автоматизации."""
        lines = [
            "#!/bin/bash",
            # Терминал запускает .command не как login-оболочку, поэтому профиль
            # приходится подключать вручную — иначе не видно ключей внешних систем.
            '[ -f "$HOME/.bash_profile" ] && . "$HOME/.bash_profile"',
            '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"',
        ]
        if log_path:
            # tee, а не перенаправление: вывод должен и сохраниться, и остаться
            # видимым в Терминале — ради него мы от встроенного журнала и ушли.
            lines.append(f"exec > >(tee {shlex.quote(str(log_path))}) 2>&1")

        lines += [
            "echo " + shlex.quote(f"Файл: {self.selected_file.name}"),
            "echo",
            " ".join(shlex.quote(str(part)) for part in cmd),
            "status=$?",
            "echo",
            "if [ $status -eq 0 ]; then echo '✅ Готово.'; "
            'else echo "❌ Завершилось с кодом $status"; fi',
            "echo 'Окно можно закрыть.'",
            'rm -f -- "$0"',
        ]

        # Уникальное имя: при параллельных запусках общий файл успевал
        # перезаписаться до того, как Терминал его прочитает, и оба окна
        # уходили распознавать одно и то же.
        fd, name = tempfile.mkstemp(prefix="ai_transcriber_", suffix=".command")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        launcher = Path(name)
        launcher.chmod(0o755)
        return launcher

    def _latest_result(self):
        """Самый свежий результат для выбранного аудио.

        Имя содержит дату и время прогона, поэтому вычислить его заранее
        нельзя — ищем по маске и берём последний по времени изменения.
        """
        if not self.selected_file:
            return None
        folder = self.selected_file.parent
        pattern = f"{self.selected_file.stem}_транскрибация*.txt"
        found = sorted(folder.glob(pattern), key=lambda f: f.stat().st_mtime)
        return found[-1] if found else None

    def open_result_file(self):
        result = self._latest_result()
        if result:
            subprocess.run(["open", str(result)])
        else:
            messagebox.showinfo(
                "Результата пока нет",
                "Файл ещё не готов — дождитесь завершения в окне Терминала.",
                parent=self.root,
            )

    def open_result_dir(self):
        target = self._latest_result() or self.selected_file
        if target and target.exists():
            subprocess.run(["open", "-R", str(target)])

def main():
    root = tk.Tk()
    app = TranscribeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
