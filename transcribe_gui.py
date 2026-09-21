import os
import sys
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

MODAL_BIN = "/Users/maksim/.local/bin/modal"
SCRIPT_PATH = "/Users/maksim/.local/bin/transcribe_modal.py"

class TranscribeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Транскрибация (WhisperX + GPU)")
        self.root.geometry("640x580")
        self.root.minsize(580, 520)

        self.selected_file = None
        self.result_file = None
        self.is_running = False

        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Общие стили
        self.root.configure(bg="#F5F5F7")
        style.configure("TFrame", background="#F5F5F7")
        style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        style.configure("TLabel", background="#F5F5F7", font=("SF Pro Text", 12))
        style.configure("Card.TLabel", background="#FFFFFF", font=("SF Pro Text", 12))
        style.configure("Header.TLabel", background="#F5F5F7", font=("SF Pro Display", 18, "bold"), foreground="#1D1D1F")
        style.configure("SubHeader.TLabel", background="#F5F5F7", font=("SF Pro Text", 11), foreground="#86868B")
        
        style.configure("Primary.TButton", font=("SF Pro Text", 13, "bold"), background="#0071E3", foreground="#FFFFFF")
        style.map("Primary.TButton", background=[("active", "#0077ED"), ("disabled", "#D2D2D7")])

    def _build_ui(self):
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)

        # 1. Заголовок
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))
        
        lbl_title = ttk.Label(header_frame, text="🎙️ Облачная AI Транскрибация", style="Header.TLabel")
        lbl_title.pack(anchor="w")
        lbl_sub = ttk.Label(header_frame, text="WhisperX + Pyannote на серверной видеокарте Nvidia A10G", style="SubHeader.TLabel")
        lbl_sub.pack(anchor="w", pady=(2, 0))

        # 2. Карточка выбора файла
        file_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        file_card.pack(fill="x", pady=(0, 12))

        lbl_file_title = ttk.Label(file_card, text="Аудио или видеофайл:", style="Card.TLabel", font=("SF Pro Text", 12, "bold"))
        lbl_file_title.pack(anchor="w", pady=(0, 8))

        file_select_box = ttk.Frame(file_card, style="Card.TFrame")
        file_select_box.pack(fill="x")

        self.btn_select = ttk.Button(file_select_box, text="Выбрать файл...", command=self.choose_file)
        self.btn_select.pack(side="left", padx=(0, 10))

        self.lbl_file_name = ttk.Label(file_select_box, text="Файл не выбран", style="Card.TLabel", foreground="#86868B")
        self.lbl_file_name.pack(side="left", fill="x", expand=True)

        # 3. Карточка параметров
        opts_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        opts_card.pack(fill="x", pady=(0, 15))

        opts_grid = ttk.Frame(opts_card, style="Card.TFrame")
        opts_grid.pack(fill="x")

        # Спикеры
        ttk.Label(opts_grid, text="Спикеры:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.spk_var = tk.StringVar(value="2")
        self.spk_combo = ttk.Combobox(opts_grid, textvariable=self.spk_var, values=["Автоопределение", "2", "3", "4", "5", "6"], width=15, state="readonly")
        self.spk_combo.grid(row=0, column=1, sticky="w", padx=(0, 25), pady=4)

        # Язык
        ttk.Label(opts_grid, text="Язык речи:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)
        self.lang_var = tk.StringVar(value="Русский (ru)")
        self.lang_combo = ttk.Combobox(opts_grid, textvariable=self.lang_var, values=["Русский (ru)", "Английский (en)", "Автоопределение"], width=15, state="readonly")
        self.lang_combo.grid(row=0, column=3, sticky="w", pady=4)

        # 4. Большая кнопка СТАРТ
        self.btn_start = tk.Button(
            main_container,
            text="🚀  Начать транскрибацию",
            font=("SF Pro Text", 14, "bold"),
            bg="#0071E3",
            fg="#FFFFFF",
            activebackground="#0077ED",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=15,
            pady=10,
            cursor="pointinghand",
            command=self.start_transcription,
            state="disabled"
        )
        self.btn_start.pack(fill="x", pady=(0, 15))

        # 5. Окно лога / Прогресс
        log_frame = ttk.Frame(main_container)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            font=("Menlo", 11),
            bg="#FFFFFF",
            fg="#1D1D1F",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=8
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "Выберите аудиофайл и нажмите «Начать транскрибацию»...\n")
        self.log_text.config(state="disabled")

        # 6. Кнопки действий после завершения
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
            size_mb = self.selected_file.stat().st_size / (1024 * 1024)
            self.lbl_file_name.config(
                text=f"{self.selected_file.name} ({size_mb:.1f} МБ)",
                foreground="#1D1D1F"
            )
            self.btn_start.config(state="normal", bg="#0071E3")
            self._log_msg(f"Выбран файл: {self.selected_file.name}\nГотов к запуску.")

    def _log_msg(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def start_transcription(self):
        if not self.selected_file or self.is_running:
            return

        self.is_running = True
        self.btn_start.config(state="disabled", bg="#D2D2D7")
        self.btn_select.config(state="disabled")
        self.btn_open_file.config(state="disabled")
        self.btn_open_dir.config(state="disabled")

        # Параметры
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

        self._log_msg("\n" + "="*45)
        self._log_msg(f"🚀 Запуск в облаке GPU Modal (A10G)...")
        self._log_msg(f"Файл: {self.selected_file.name}")
        self._log_msg(f"Параметры: Язык = {lang_code}, Спикеры = {spk_val}")
        self._log_msg("="*45 + "\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd):
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            for line in iter(process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    self.root.after(0, self._log_msg, clean_line)

            process.stdout.close()
            ret_code = process.wait()

            self.root.after(0, self._on_finished, ret_code)
        except Exception as e:
            self.root.after(0, self._log_msg, f"Ошибка выполнения: {e}")
            self.root.after(0, self._on_finished, 1)

    def _on_finished(self, ret_code):
        self.is_running = False
        self.btn_start.config(state="normal", bg="#0071E3")
        self.btn_select.config(state="normal")

        if ret_code == 0:
            self.result_file = self.selected_file.with_name(f"{self.selected_file.stem}_транскрибация.txt")
            self.btn_open_file.config(state="normal")
            self.btn_open_dir.config(state="normal")

            # Звук завершения
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True)
            # Уведомление macOS
            os.system(f'osascript -e \'display notification "Файл успешно расшифрован и разделен по ролям!" with title "AI Транскрибатор"\'')
            messagebox.showinfo("Готово!", f"Транскрибация завершена успешно!\nРезультат сохранен в:\n{self.result_file.name}")
        else:
            messagebox.showerror("Ошибка", "Произошла ошибка при выполнении. Подробности в окне лога.")

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
