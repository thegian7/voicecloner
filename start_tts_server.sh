#!/usr/bin/env bash
# Start the TTS server for Claude Code voice output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Start TTS server
echo "Starting VoxCPM2 TTS server for Claude Code..."
python tts_server.py "$@"
