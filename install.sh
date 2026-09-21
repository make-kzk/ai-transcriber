#!/bin/bash
set -e

echo "🚀 Установка AI Transcriber на macOS..."

# 1. Проверка или установка uv
if ! command -v uv &> /dev/null; then
    echo "Установка пакетного менеджера uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Создание изолированного виртуального окружения
VENV_DIR="$HOME/.modal-venv"
echo "Создание Python окружения в $VENV_DIR..."
uv venv --python 3.11 "$VENV_DIR" --clear
uv pip install --python "$VENV_DIR/bin/python" modal

# 3. Копирование скриптов в ~/.local/bin
mkdir -p "$HOME/.local/bin"
cp transcribe_modal.py "$HOME/.local/bin/transcribe_modal.py"
cp transcribe_gui.py "$HOME/.local/bin/transcribe_gui.py"
cp transcribe "$HOME/.local/bin/transcribe"
chmod +x "$HOME/.local/bin/transcribe"
ln -sf "$VENV_DIR/bin/modal" "$HOME/.local/bin/modal"

# Добавление в PATH
for RC in "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc"; do
    if [ -f "$RC" ]; then
        grep -q ".local/bin" "$RC" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
    fi
done

# 4. Сборка .app бандла
APP_DIR="/Applications/AI Transcriber.app"
echo "Сборка приложения $APP_DIR..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

if [ -f "/System/Applications/VoiceMemos.app/Contents/Resources/MacAppIcon.icns" ]; then
    cp "/System/Applications/VoiceMemos.app/Contents/Resources/MacAppIcon.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"
fi

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
    <string>1.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cat << 'EOF' > "$APP_DIR/Contents/MacOS/AI Transcriber"
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
exec "$HOME/.modal-venv/bin/python" "$HOME/.local/bin/transcribe_gui.py"
EOF

chmod +x "$APP_DIR/Contents/MacOS/AI Transcriber"

# Копия на Рабочий стол
rm -rf "$HOME/Desktop/AI Transcriber.app"
cp -R "$APP_DIR" "$HOME/Desktop/AI Transcriber.app"

echo "✅ Установка завершена!"
echo "Приложение доступно в /Applications и на Рабочем столе."
echo "Команда терминала: transcribe <файл>"
