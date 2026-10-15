# VoiceCloner

Clone your voice locally with [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). No data leaves your machine.

Record a few seconds of your voice, then generate speech in your voice from any text. Includes a full audiobook production mode for long-form content.

## Features

- **Quick Clone** -- Record or upload 5-10 seconds of your voice, type text, get speech output
- **Audiobook Mode** -- Load an EPUB or paste text, chunk it into chapters, generate and review audio per-chunk, export a stitched audiobook
- **Voice Library** -- Save and manage multiple cloned voices
- **Style prompts** -- Natural-language voice descriptions ("warm energetic podcast host") that condition the clone
- **Export** -- WAV, MP3, FLAC, M4A output formats (48 kHz studio quality)
- **Runs locally** -- All inference happens on your machine (MPS/CUDA/CPU)

## Quick Start (Python)

Requires Python 3.10-3.12 (VoxCPM2 doesn't support 3.13 yet).

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

`tts_server.py` exposes an OpenAI-compatible `/v1/audio/speech` endpoint using your cloned voice (powered by VoxCPM2). Useful for integrating with other tools.

```bash
source .venv/bin/activate
python tts_server.py --voice "MyVoice" --port 8765
```

## Speaker LoRA (fine-tuned voices)

A saved voice can carry a LoRA fine-tuned on that speaker with VoxCPM's own trainer
(`scripts/train_voxcpm_finetune.py` in the VoxCPM repo, `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`).
A few hours of the speaker's natural speech gives a voice that is noticeably closer than a
clone from one reference clip. Attach the checkpoint to a saved voice:

```bash
python -c 'from core.voices import attach_lora; attach_lora("MyVoice", "path/to/ckpt/step_0001000")'
```

That copies `lora_weights.safetensors` and `lora_config.json` into
`~/.voicecloner/voices/MyVoice/lora/`. From then on, anything generated from that voice's
reference recording uses the LoRA, in the app, the TTS server, audiobooks and the CLI. The
first LoRA voice in a session rebuilds the model with LoRA layers (once, ~30 s); other voices
switch the LoRA off. Pass `lora_path=""` to `create_clone_prompt` to force a plain clone.

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

Models are downloaded from Hugging Face on first run (~5GB).

## License

[MIT](LICENSE)
