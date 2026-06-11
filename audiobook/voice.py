"""Voice setup logic for audiobook production."""

from __future__ import annotations

import tempfile

import soundfile as sf
import gradio as gr

from core import ModelCache, create_clone_prompt, generate_preview, save_wav
from .state import TEMP_DIR


cache = ModelCache()


def setup_voice(
    ref_audio,
    ref_text: str,
    style: str,
    use_design: bool,
) -> tuple:
    """Build voice clone prompt from reference audio.

    Returns (voice_prompt, status_text).
    """
    if ref_audio is None:
        raise gr.Error("Please upload reference audio.")

    if isinstance(ref_audio, tuple):
        sr_in, audio_data = ref_audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR)
        sf.write(tmp.name, audio_data, sr_in)
        ref_audio_path = tmp.name
    else:
        ref_audio_path = ref_audio

    style_arg = style if (use_design and style.strip()) else ""
    prompt = create_clone_prompt(cache.clone, ref_audio_path, ref_text, style=style_arg)
    return prompt, "Voice ready. You can preview or proceed to generation."


def preview_voice(voice_prompt, preview_text: str) -> str | None:
    """Generate a short preview clip and return the file path."""
    if voice_prompt is None:
        raise gr.Error("Set up your voice first.")
    text = preview_text.strip() or "Hello! This is a preview of your cloned voice."
    audio, sr = generate_preview(text, cache.clone, voice_prompt)
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR).name
    save_wav(path, audio, sr)
    return path
