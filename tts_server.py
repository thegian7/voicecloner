#!/usr/bin/env python3
"""
TTS API Server for Claude Code Voice Integration

Exposes an OpenAI-compatible /v1/audio/speech endpoint that uses Qwen3-TTS
with your cloned voice. Plays audio through system speakers.

Usage:
    python tts_server.py                    # Default voice (reference.wav)
    python tts_server.py --voice "MyVoice"  # Use saved voice by name
    python tts_server.py --port 8000        # Custom port

Then configure Claude Code hook:
    export TTS_URL="http://localhost:8765/v1/audio/speech"
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Add parent dir to path for core module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    ModelCache,
    run_voice_design,
    create_clone_prompt,
    generate_full,
    save_wav,
    load_voice,
    list_voices,
)

# Global state
cache = None
voice_prompt = None
current_voice_name = None

DEFAULT_STYLE = (
    "A young excited American male with a warm energetic tone, "
    "speaking with enthusiasm and confidence like a podcast host"
)

# Default reference audio (fallback)
DEFAULT_REF_AUDIO = os.path.join(os.path.dirname(__file__), "reference_excited.wav")
DEFAULT_REF_TEXT = "Hello! What up everybody, hope you are having an awesome day!"


def play_audio(wav_path: str):
    """Play audio through system speakers (macOS/Linux)."""
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == "linux":
        subprocess.Popen(["aplay", "-q", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"Audio saved to {wav_path} (auto-play not supported on this platform)")


def init_voice(voice_name: str = None, ref_audio: str = None, ref_text: str = None, style: str = None, fast_mode: bool = True):
    """Initialize the TTS model with a voice."""
    global cache, voice_prompt, current_voice_name

    print("Loading Qwen3-TTS model (fast mode)..." if fast_mode else "Loading Qwen3-TTS models...")

    # Fast mode: skip VoiceDesign, use float16
    if fast_mode:
        import torch
        from qwen_tts import Qwen3TTSModel
        from core.device import get_device

        device = get_device()
        print(f"Loading Clone model on {device} with SDPA attention...")

        # Load clone model with SDPA attention for faster inference
        # (float32 required for MPS stability, float16 causes NaN)
        clone_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=torch.float32,
            attn_implementation="sdpa"  # Use PyTorch's scaled dot-product attention
        )

        # Optionally compile the model for faster inference (requires PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and device != "mps":  # torch.compile has issues on MPS
                print("Compiling model with torch.compile...")
                clone_model.model = torch.compile(clone_model.model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")

        # Create a simple object to hold the model
        class FastCache:
            def __init__(self, clone):
                self.clone = clone
        cache = FastCache(clone_model)
    else:
        cache = ModelCache()

    if voice_name:
        # Load saved voice
        voices = [v.name for v in list_voices()]
        if voice_name not in voices:
            print(f"Voice '{voice_name}' not found. Available: {voices}")
            sys.exit(1)

        v = load_voice(voice_name)
        ref_audio = v.ref_audio_path
        ref_text = v.ref_text
        style = v.style
        use_design = v.use_design and not fast_mode
        current_voice_name = voice_name
        print(f"Loaded voice: {voice_name}")
    else:
        # Use provided or default reference
        ref_audio = ref_audio or DEFAULT_REF_AUDIO
        ref_text = ref_text or DEFAULT_REF_TEXT
        style = style or DEFAULT_STYLE
        use_design = not fast_mode
        current_voice_name = "default"
        print(f"Using reference audio: {ref_audio}")

    if not os.path.exists(ref_audio):
        print(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    # Run voice design if enabled (skipped in fast mode)
    if use_design and style and hasattr(cache, 'design'):
        print(f"Applying voice style: {style[:50]}...")
        run_voice_design(cache.design, style)

    # Create clone prompt
    print("Building voice clone prompt...")
    voice_prompt = create_clone_prompt(cache.clone, ref_audio, ref_text)
    print("Voice ready!")


class TTSHandler(BaseHTTPRequestHandler):
    """Handle TTS API requests."""

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def do_POST(self):
        if self.path == "/v1/audio/speech":
            self.handle_speech()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "voice": current_voice_name}).encode())
        elif self.path == "/voices":
            voices = [v.name for v in list_voices()]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"voices": voices}).encode())
        else:
            self.send_error(404)

    def handle_speech(self):
        global cache, voice_prompt

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        text = data.get("input", "")
        if not text.strip():
            self.send_error(400, "No input text provided")
            return

        print(f"TTS: {text[:80]}...")

        try:
            # Generate audio
            audio, sr = generate_full(text, cache.clone, voice_prompt)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            save_wav(wav_path, audio, sr)

            # Play audio in background thread
            threading.Thread(target=play_audio, args=(wav_path,), daemon=True).start()

            # Return audio data
            with open(wav_path, "rb") as f:
                audio_data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio_data)))
            self.end_headers()
            self.wfile.write(audio_data)

        except Exception as e:
            print(f"TTS error: {e}")
            self.send_error(500, str(e))


def main():
    parser = argparse.ArgumentParser(description="TTS API Server for Claude Code")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--voice", type=str, help="Name of saved voice to use")
    parser.add_argument("--ref", type=str, help="Path to reference audio file")
    parser.add_argument("--ref-text", type=str, help="Transcript of reference audio")
    parser.add_argument("--style", type=str, help="Voice style description")
    parser.add_argument("--list-voices", action="store_true", help="List saved voices and exit")
    args = parser.parse_args()

    if args.list_voices:
        voices = list_voices()
        if not voices:
            print("No saved voices. Create one using the VoiceCloner app.")
        else:
            print("Saved voices:")
            for v in voices:
                print(f"  - {v.name}")
        return

    # Initialize voice
    init_voice(
        voice_name=args.voice,
        ref_audio=args.ref,
        ref_text=args.ref_text,
        style=args.style,
    )

    # Start server
    server = HTTPServer(("0.0.0.0", args.port), TTSHandler)
    print(f"\nTTS Server running on http://localhost:{args.port}")
    print(f"Endpoint: http://localhost:{args.port}/v1/audio/speech")
    print("\nTo use with Claude Code, set:")
    print(f'  export TTS_URL="http://localhost:{args.port}/v1/audio/speech"')
    print("\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
