import os
import sys
import shlex
import shutil
import tempfile
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
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
        main_container = ttk.Frame(self.root, padding=20)
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
