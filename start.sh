#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== VoiceCloner ==="

# Check Python
if command -v python3.13 &>/dev/null; then
    PYTHON=python3.13
elif command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "Error: Python 3.12+ required. Install from https://python.org"
    exit 1
fi

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies if needed
if ! python -c "import qwen_tts" &>/dev/null; then
    echo "Installing dependencies (first run only)..."
    pip install -U qwen-tts soundfile
fi

echo "Starting VoiceCloner at http://localhost:7860"
python "$SCRIPT_DIR/app.py"
