# VoiceCloner

Clone your voice locally with [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS). No data leaves your machine.

Record a few seconds of your voice, then generate speech in your voice from any text. Includes a full audiobook production mode for long-form content.

## Features

- **Quick Clone** -- Record or upload 5-10 seconds of your voice, type text, get speech output
- **Audiobook Mode** -- Load an EPUB or paste text, chunk it into chapters, generate and review audio per-chunk, export a stitched audiobook
- **Voice Library** -- Save and manage multiple cloned voices
- **VoiceDesign** -- Optional style descriptions to control tone and delivery
- **Export** -- WAV, MP3, FLAC, M4A output formats
- **Runs locally** -- All inference happens on your machine (MPS/CUDA/CPU)

## Quick Start (Python)

Requires Python 3.12+.

```bash
git clone https://github.com/thegian7/voicecloner.git
cd voicecloner
./start.sh
```

`start.sh` creates a virtual environment, installs dependencies, and launches the Gradio UI at `http://localhost:7860`.

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Desktop App (Electron)

The Electron shell wraps the Gradio UI into a native desktop app with automatic Python/venv setup and GPU detection.

```bash
npm install
npm run dist:mac    # or dist:win / dist:linux
```

This bundles a standalone Python runtime (downloaded via `npm run download-python`) so end users don't need Python installed.

### Build targets

| Platform | Format | GPU Support |
|----------|--------|-------------|
| macOS | DMG (arm64 + x64) | MPS (Apple Silicon) |
| Windows | NSIS installer (x64) | CUDA, CPU |
| Linux | AppImage (x64) | CUDA, ROCm, CPU |

## TTS API Server

`tts_server.py` exposes an OpenAI-compatible `/v1/audio/speech` endpoint using your cloned voice. Useful for integrating with other tools.

```bash
source .venv/bin/activate
python tts_server.py --voice "MyVoice" --port 8765
```

## Project Structure

```
app.py                  # Main Gradio app (Quick Clone + Audiobook + Voices)
core/                   # Shared TTS engine (model loading, generation, audio processing)
audiobook/              # Audiobook-specific logic (chapters, export, state)
electron/               # Electron desktop shell
python/                 # Python files bundled into Electron builds
tts_server.py           # Standalone TTS API server
start.sh                # One-command launcher
```

## Hardware Requirements

- **Apple Silicon Mac**: Works out of the box via MPS. 16GB RAM recommended.
- **NVIDIA GPU**: CUDA 12.4+ with 8GB+ VRAM recommended.
- **AMD GPU**: ROCm 6.2+ supported.
- **CPU**: Works but slow. 32GB RAM recommended.

Models are downloaded from Hugging Face on first run (~2GB).

## License

[MIT](LICENSE)
