#!/usr/bin/env python3
"""
TTS API Server for Claude Code Voice Integration

Exposes an OpenAI-compatible /v1/audio/speech endpoint that uses VoxCPM2
with your cloned voice. Streams audio directly to ffplay/sox for low-latency
local playback — no WAV file on disk, no buffering the full waveform.

Usage:
    python tts_server.py                    # Default voice (reference.wav)
    python tts_server.py --voice "MyVoice"  # Use saved voice by name
    python tts_server.py --port 8000        # Custom port
    python tts_server.py --no-autoplay      # Headless mode (no local playback)

Then configure Claude Code hook:
    export TTS_URL="http://localhost:8765/v1/audio/speech"
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import soundfile as sf

# Add parent dir to path for core module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    ModelCache,
    create_clone_prompt,
    generate_stream,
    load_voice,
    list_voices,
)

# Global state
cache = None
voice_prompt = None
current_voice_name = None
autoplay_enabled = True
player_bin = None

DEFAULT_STYLE = ""  # Reference audio carries the voice; long style descriptions
                    # leak into the spoken output. Use VoiceDesign mode (no ref)
                    # if you want a styled non-cloned voice.

# Default reference audio (fallback)
DEFAULT_REF_AUDIO = os.path.join(os.path.dirname(__file__), "reference_excited.wav")
DEFAULT_REF_TEXT = "Hello! What up everybody, hope you are having an awesome day!"

# OpenAI /v1/audio/speech response_format -> (content-type, ffmpeg/pydub format)
_AUDIO_FORMATS = {
    "wav": ("audio/wav", None),
    "mp3": ("audio/mpeg", "mp3"),
    "opus": ("audio/ogg", "opus"),
    "aac": ("audio/aac", "adts"),
    "flac": ("audio/flac", "flac"),
    "ogg": ("audio/ogg", "ogg"),
    "pcm": ("audio/wav", None),
}


def _encode_audio(audio: "np.ndarray", sr: int, fmt: str) -> tuple[bytes, str]:
    """Encode a float waveform to the requested container. Returns (bytes, content_type)."""
    fmt = (fmt or "mp3").lower()
    content_type, ff_fmt = _AUDIO_FORMATS.get(fmt, _AUDIO_FORMATS["mp3"])

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()
    if ff_fmt is None:  # wav / pcm — return as-is
        return wav_bytes, content_type

    # Transcode via ffmpeg (installed) for compressed formats.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", ff_fmt, "pipe:1"],
        input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        # Fall back to WAV rather than failing the request.
        return wav_bytes, "audio/wav"
    return proc.stdout, content_type


def _find_player() -> str | None:
    """Find a pipe-capable audio player. ffplay preferred, then sox 'play'."""
    for cmd in ("ffplay", "play"):
        path = shutil.which(cmd)
        if path:
            return path
    return None


def _player_args(player: str, sample_rate: int) -> list[str]:
    """Build CLI args to pipe raw float32 mono PCM into the given player."""
    if player.endswith("ffplay"):
        # ffmpeg 8.x renamed -ac to -ch_layout. Use the new spelling.
        return [
            player,
            "-f", "f32le", "-ar", str(sample_rate), "-ch_layout", "mono",
            "-nodisp", "-autoexit", "-loglevel", "error",
            "-i", "-",
        ]
    if player.endswith("play"):
        # sox play
        return [
            player,
            "-q",
            "-t", "raw", "-r", str(sample_rate), "-b", "32", "-e", "float", "-c", "1",
            "-",
        ]
    raise RuntimeError(f"Unsupported player: {player}")


def stream_to_player(chunk_iter, sample_rate: int) -> int:
    """Pipe audio chunks to the configured player. Returns total samples written."""
    proc = subprocess.Popen(
        _player_args(player_bin, sample_rate),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=None,  # let player errors surface in server logs
    )
    total = 0
    try:
        for chunk, _ in chunk_iter:
            try:
                proc.stdin.write(chunk.astype("float32").tobytes())
                proc.stdin.flush()
            except BrokenPipeError:
                # Player exited (user killed it, autoexit fired, etc.) — stop.
                break
            total += len(chunk)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    return total


def init_voice(voice_name: str = None, ref_audio: str = None, ref_text: str = None, style: str = None):
    """Initialize the TTS model and build the voice prompt."""
    global cache, voice_prompt, current_voice_name

    cache = ModelCache()

    if voice_name:
        voices = [v.name for v in list_voices()]
        if voice_name not in voices:
            print(f"Voice '{voice_name}' not found. Available: {voices}")
            sys.exit(1)

        v = load_voice(voice_name)
        ref_audio = v.ref_audio_path
        ref_text = v.ref_text
        # Saved voices may have a style, but with a reference audio the style
        # gets read aloud rather than acting as a cue. Honor explicit override
        # via --style, otherwise leave empty.
        style = style if style is not None else ""
        current_voice_name = voice_name
        print(f"Loaded voice: {voice_name}")
    else:
        ref_audio = ref_audio or DEFAULT_REF_AUDIO
        ref_text = ref_text or DEFAULT_REF_TEXT
        style = style if style is not None else DEFAULT_STYLE
        current_voice_name = "default"
        print(f"Using reference audio: {ref_audio}")

    if not os.path.exists(ref_audio):
        print(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    # Touch the lazy property so the model loads now (not on first request).
    _ = cache.tts
    voice_prompt = create_clone_prompt(cache.tts, ref_audio, ref_text, style=style)
    print("Voice ready!")


class TTSHandler(BaseHTTPRequestHandler):
    """Handle TTS API requests."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == "/v1/audio/speech":
            self.handle_speech()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "voice": current_voice_name, "player": player_bin})
        elif self.path == "/voices":
            self._json(200, {"voices": [v.name for v in list_voices()]})
        else:
            self.send_error(404)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_speech(self):
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
            chunk_iter = generate_stream(text, cache.tts, voice_prompt)
            sr = int(cache.tts.tts_model.sample_rate)

            if autoplay_enabled and player_bin:
                samples = stream_to_player(chunk_iter, sr)
                self._json(200, {
                    "status": "ok",
                    "samples": samples,
                    "duration": round(samples / sr, 2),
                    "sample_rate": sr,
                })
            else:
                # Headless / server mode: buffer the full waveform and return it
                # as encoded audio bytes (OpenAI /v1/audio/speech contract), so
                # remote callers like Hermes can attach it to a voice message.
                parts = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk, _ in chunk_iter]
                audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                fmt = (data.get("response_format") or "mp3").lower()
                payload, content_type = _encode_audio(audio, sr, fmt)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except Exception as e:
            print(f"TTS error: {e}")
            self.send_error(500, str(e))


def main():
    parser = argparse.ArgumentParser(description="TTS API Server for Claude Code")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--voice", type=str, help="Name of saved voice to use")
    parser.add_argument("--ref", type=str, help="Path to reference audio file")
    parser.add_argument("--ref-text", type=str, help="Transcript of reference audio")
    parser.add_argument("--style", type=str, help="Voice style description (only honored with no reference audio)")
    parser.add_argument("--list-voices", action="store_true", help="List saved voices and exit")
    parser.add_argument("--no-autoplay", action="store_true", help="Don't play audio locally (headless / server mode)")
    args = parser.parse_args()

    global autoplay_enabled, player_bin
    autoplay_enabled = not args.no_autoplay
    player_bin = _find_player()

    if autoplay_enabled and not player_bin:
        print("Warning: no audio player found (need ffplay or sox 'play').")
        print("  brew install ffmpeg   # for ffplay")
        print("  brew install sox      # for play")
        print("Falling back to headless mode.")
        autoplay_enabled = False

    if args.list_voices:
        voices = list_voices()
        if not voices:
            print("No saved voices. Create one using the VoiceCloner app.")
        else:
            print("Saved voices:")
            for v in voices:
                print(f"  - {v.name}")
        return

    init_voice(
        voice_name=args.voice,
        ref_audio=args.ref,
        ref_text=args.ref_text,
        style=args.style,
    )

    server = HTTPServer(("0.0.0.0", args.port), TTSHandler)
    mode = "streaming via " + player_bin if autoplay_enabled else "headless"
    print(f"\nTTS Server running on http://localhost:{args.port} ({mode})")
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
