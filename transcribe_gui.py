import os
import sys
import re
import pty
import time
import fcntl
import codecs
import struct
import shutil
import termios
import subprocess
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Управляющие последовательности, которыми rich рисует живой вывод modal:
# CSI (цвета, перемещение курсора), OSC (заголовок окна) и одиночные Esc-коды.
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)

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
        self.result_file = None
        self.is_running = False
        self.start_time = None
        self.timer_id = None
        self.current_process = None
        self.was_cancelled = False
        self.heartbeat_id = None
        self.last_output_ts = None

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

        # 4. Панель кнопок: СТАРТ и ОТМЕНА
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
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_cancel = tk.Button(
            btn_box,
            text="🛑 Отмена",
            font=("SF Pro Text", 13),
            bg="#FF3B30",
            fg="#FFFFFF",
            activebackground="#D32F2F",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=12,
            pady=8,
            cursor="pointinghand",
            command=self.cancel_transcription,
            state="disabled",
        )
        self.btn_cancel.pack(side="right")

        # 5. Карточка статуса и прогресса
        status_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        status_card.pack(fill="x", pady=(0, 10))

        status_header = ttk.Frame(status_card, style="Card.TFrame")
        status_header.pack(fill="x", pady=(0, 6))

        self.lbl_status = ttk.Label(status_header, text="Ожидание файла", style="Status.TLabel")
        self.lbl_status.pack(side="left")

        self.lbl_timer = ttk.Label(status_header, text="00:00", style="Timer.TLabel")
        self.lbl_timer.pack(side="right")

        self.progress_bar = ttk.Progressbar(status_card, mode="indeterminate", length=400)
        self.progress_bar.pack(fill="x", pady=(4, 6))

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

    def _on_proc_line(self, text):
        """Строка пришла от процесса — сбрасываем таймер тишины."""
        self.last_output_ts = time.time()
        self._log_msg(text)

    def _heartbeat(self):
        """Показывает, что работа идёт, даже когда modal молчит.

        Вывод дочернего процесса буферизуется и может не доходить до окна
        минутами; без этого журнал выглядит застывшим и неотличим от зависания.
        """
        if not self.is_running:
            return
        quiet_for = time.time() - (self.last_output_ts or self.start_time or time.time())
        if quiet_for >= 15:
            elapsed = int(time.time() - self.start_time) if self.start_time else 0
            m, s = divmod(elapsed, 60)
            stage = self.lbl_status.cget("text")
            self._log_msg(f"... {m:02d}:{s:02d} — работа идёт ({stage}); "
                          f"новых сообщений нет {int(quiet_for)} с")
        self.heartbeat_id = self.root.after(15000, self._heartbeat)

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

    def _update_timer(self):
        if self.is_running and self.start_time:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            self.lbl_timer.config(text=f"{m:02d}:{s:02d}")
            self.timer_id = self.root.after(1000, self._update_timer)

    def start_transcription(self):
        if not self.selected_file or self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()
        self.progress_bar.start(10)
        self._update_timer()

        self.btn_start.config(text="⏳  Идет обработка в облаке...", state="disabled", bg="#D2D2D7")
        self.btn_cancel.config(state="normal")
        self.btn_select.config(state="disabled")
        self.btn_open_file.config(state="disabled")
        self.btn_open_dir.config(state="disabled")
        self.lbl_status.config(text="🚀 Запуск облачной видеокарты A10G...", foreground="#0071E3")

        spk_val = self.spk_var.get()
        speakers_arg = []
        if spk_val.isdigit():
            speakers_arg = ["--speakers", spk_val]

        lang_val = self.lang_var.get()
        if "ru" in lang_val:
            lang_code = "ru"
        elif "en" in lang_val:
            lang_code = "en"
        else:
            lang_code = "auto"

        cmd = [
            MODAL_BIN,
            "run",
            SCRIPT_PATH,
            "--file",
            str(self.selected_file),
            "--language",
            lang_code,
        ] + speakers_arg

        self._log_msg("\n" + "="*50)
        self._log_msg(f"🚀 Старт: {self.selected_file.name}")
        self._log_msg(f"Параметры: Язык = {lang_code}, Спикеры = {spk_val}")
        self._log_msg("="*50 + "\n")

        self.last_output_ts = time.time()
        self._heartbeat()

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def cancel_transcription(self):
        if self.current_process and self.is_running:
            self.was_cancelled = True
            self._log_msg("\n🛑 Отмена процесса пользователем...")
            try:
                self.current_process.terminate()
                self.root.after(2000, self._force_kill_if_needed)
            except Exception as e:
                self._log_msg(f"Ошибка при отмене: {e}")

    def _force_kill_if_needed(self):
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.kill()
            except Exception:
                pass

    def _run_process(self, cmd):
        """Запускает modal в псевдотерминале.

        Через обычный pipe modal видит !isatty() и переключается в тихий
        режим — в журнал не попадает почти ничего. С pty он выдаёт тот же
        живой вывод, что и в терминале, а ANSI-последовательности мы
        вычищаем сами.
        """
        master_fd = None
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = "200"

            master_fd, slave_fd = pty.openpty()
            try:
                # Широкое окно, чтобы rich не рвал строки на середине.
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ,
                            struct.pack("HHHH", 50, 200, 0, 0))
            except Exception:
                pass

            self.current_process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                close_fds=True,
            )
            os.close(slave_fd)

            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            buffer = ""
            last_line = None

            def emit(raw):
                nonlocal last_line
                line = ANSI_RE.sub("", raw).strip()
                # rich перерисовывает одни и те же строки — не дублируем их.
                if line and line != last_line:
                    last_line = line
                    self.root.after(0, self._on_proc_line, line)

            while True:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break  # EIO — дочерний процесс закрыл терминал
                if not data:
                    break
                for ch in decoder.decode(data):
                    if ch in ("\r", "\n"):
                        emit(buffer)
                        buffer = ""
                    else:
                        buffer += ch

            emit(buffer)
            ret_code = self.current_process.wait()
            self.root.after(0, self._on_finished, ret_code)
        except Exception as e:
            self.root.after(0, self._log_msg, f"Ошибка выполнения: {e}")
            self.root.after(0, self._on_finished, 1)
        finally:
            if master_fd is not None:
                try:
                    os.close(master_fd)
                except Exception:
                    pass

    def _on_finished(self, ret_code):
        self.is_running = False
        self.progress_bar.stop()
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        if self.heartbeat_id:
            self.root.after_cancel(self.heartbeat_id)
            self.heartbeat_id = None

        self.btn_start.config(text="🚀  Начать транскрибацию", state="normal", bg="#0071E3")
        self.btn_cancel.config(state="disabled")
        self.btn_select.config(state="normal")
        self.current_process = None

        if self.was_cancelled:
            self.lbl_status.config(text="🛑 Транскрибация отменена", foreground="#FF9500")
            self._log_msg("🛑 Транскрибация была отменена пользователем.")
            self.was_cancelled = False
            return

        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        em, es = divmod(elapsed, 60)

        if ret_code == 0:
            self.result_file = self.selected_file.with_name(f"{self.selected_file.stem}_транскрибация.txt")
            self.lbl_status.config(text=f"✅ Готово! (за {em:02d}:{es:02d})", foreground="#34C759")
            self.btn_open_file.config(state="normal")
            self.btn_open_dir.config(state="normal")

            try:
                subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True)
                os.system(f'osascript -e \'display notification "Транскрибация успешно завершена за {em:02d}:{es:02d}!" with title "AI Транскрибатор"\'')
            except Exception:
                pass
            messagebox.showinfo("Готово!", f"Транскрибация завершена успешно за {em:02d}:{es:02d}!\nРезультат сохранен рядом с аудиофайлом:\n{self.result_file.name}", parent=self.root)
        else:
            self.lbl_status.config(text="❌ Ошибка выполнения (см. журнал)", foreground="#FF3B30")
            try:
                subprocess.run(["afplay", "/System/Library/Sounds/Basso.aiff"], capture_output=True)
            except Exception:
                pass

    def open_result_file(self):
        if self.result_file and self.result_file.exists():
            subprocess.run(["open", str(self.result_file)])

    def open_result_dir(self):
        if self.result_file and self.result_file.exists():
            subprocess.run(["open", "-R", str(self.result_file)])
        elif self.selected_file and self.selected_file.exists():
            subprocess.run(["open", "-R", str(self.selected_file)])

def main():
    root = tk.Tk()
    app = TranscribeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
