#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🚀 Установка AI Transcriber на macOS из $SCRIPT_DIR..."

# 1. Проверка или установка uv
if ! command -v uv &> /dev/null; then
    echo "Установка пакетного менеджера uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Создание изолированного виртуального окружения Python 3.11
VENV_DIR="$HOME/.modal-venv"
echo "Настройка Python окружения в $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.11 "$VENV_DIR"
fi
# praat-parselmouth — измерение тона и громкости в модуле анализа речи.
# Praat, а не самодельная обработка сигнала; один пакет без тяжёлых зависимостей.
# google-genai — разбор по брифу на вкладке анализа (нужен ключ Google AI Studio).
uv pip install --python "$VENV_DIR/bin/python" modal numpy httpx praat-parselmouth google-genai

# 3. Симлинки в ~/.local/bin (чтобы изменения в коде сразу работали везде)
mkdir -p "$HOME/.local/bin"
ln -sf "$SCRIPT_DIR/transcribe_modal.py" "$HOME/.local/bin/transcribe_modal.py"
ln -sf "$SCRIPT_DIR/transcribe_gui.py" "$HOME/.local/bin/transcribe_gui.py"
ln -sf "$SCRIPT_DIR/transcribe" "$HOME/.local/bin/transcribe"
chmod +x "$SCRIPT_DIR/transcribe"
ln -sf "$VENV_DIR/bin/modal" "$HOME/.local/bin/modal"

# Добавление ~/.local/bin в PATH для всех шеллов
for RC in "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc"; do
    if [ -f "$RC" ]; then
        grep -q ".local/bin" "$RC" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
    fi
done

# 4. Сборка и подпись .app бандла
APP_DIR="/Applications/AI Transcriber.app"
echo "Сборка приложения $APP_DIR..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Иконка
if [ -f "/System/Applications/VoiceMemos.app/Contents/Resources/MacAppIcon.icns" ]; then
    cp "/System/Applications/VoiceMemos.app/Contents/Resources/MacAppIcon.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"
fi

# Info.plist с TCC-разрешениями для доступа к папкам Mac
cat << 'EOF' > "$APP_DIR/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AI Transcriber</string>
    <key>CFBundleIdentifier</key>
    <string>com.maksim.aitranscriber</string>
    <key>CFBundleName</key>
    <string>AI Transcriber</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSDocumentsFolderUsageDescription</key>
    <string>AI Transcriber требуется доступ к папке Документы для выбора и сохранения аудиозаписей.</string>
    <key>NSDownloadsFolderUsageDescription</key>
    <string>AI Transcriber требуется доступ к папке Загрузки для выбора аудиофайлов.</string>
    <key>NSDesktopFolderUsageDescription</key>
    <string>AI Transcriber требуется доступ к Рабочему столу для выбора аудиофайлов.</string>
</dict>
</plist>
EOF

# Без exec: процесс, запущенный LaunchServices, должен остаться живым,
# иначе TCC читает подпись у подменённого образа (python) вместо бандла.
cat << 'EOF' > "$APP_DIR/Contents/MacOS/AI Transcriber"
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
"$HOME/.modal-venv/bin/python" "$HOME/.local/bin/transcribe_gui.py"
EOF

chmod +x "$APP_DIR/Contents/MacOS/AI Transcriber"

# Ad-hoc подпись интерпретатора: uv ставит python без подписи вообще
# ("code object is not signed at all"), поэтому macOS не может привязать
# к нему разрешения и отказывает в доступе к Рабочему столу, Документам
# и Загрузкам с ошибкой PermissionError(1, 'Operation not permitted').
BASE_PYTHON="$("$VENV_DIR/bin/python" -c 'import sys; print(sys._base_executable or sys.executable)')"
if [ -n "$BASE_PYTHON" ] && [ -f "$BASE_PYTHON" ]; then
    echo "Подпись интерпретатора $BASE_PYTHON..."
    codesign --force --sign - "$BASE_PYTHON" 2>/dev/null || \
        echo "  Предупреждение: подписать интерпретатор не удалось."
fi

# Ad-hoc цифровая подпись бандла (для авторизации в macOS TCC)
echo "Цифровая подпись бандла macOS (codesign)..."
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true

# Копия на Рабочий стол
rm -rf "$HOME/Desktop/AI Transcriber.app"
cp -R "$APP_DIR" "$HOME/Desktop/AI Transcriber.app"
codesign --force --deep --sign - "$HOME/Desktop/AI Transcriber.app" 2>/dev/null || true

echo "✅ Установка и настройка успешно завершены!"
echo "Приложение доступно в /Applications и на Рабочем столе."
echo "Команда терминала: transcribe <файл>"
