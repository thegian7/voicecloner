from typing import Callable

import numpy as np

from .text import prepare_text
from .audio import stitch_chunks


def run_voice_design(design_model, style: str) -> None:
    """Run VoiceDesign to condition the model on a style."""
    print(f"Generating voice style: {style[:60]}...")
    design_model.generate_voice_design(
        text="Hello everybody, hope you are having an awesome day!",
        language="English",
        instruct=style,
    )


def create_clone_prompt(clone_model, ref_audio: str, ref_text: str = ""):
    """Build a voice clone prompt from reference audio."""
    print("Building voice clone prompt...")
    if ref_text.strip():
        return clone_model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text
        )
    return clone_model.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text="", x_vector_only_mode=True
    )


def generate_single(clone_model, text: str, voice_clone_prompt) -> tuple[np.ndarray, int]:
    """Generate audio for a single text chunk."""
    wavs, sr = clone_model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    return wavs[0], sr


def generate_chunked(
    chunks: list[str],
    clone_model,
    voice_clone_prompt,
    on_chunk: Callable[[int, int, np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Generate audio per chunk and stitch with crossfades.

    on_chunk(index, total, audio, sr) is called after each chunk if provided.
    """
    audio_list: list[np.ndarray] = []
    paragraph_flags: list[bool] = []
    sr: int = 0

    for i, chunk in enumerate(chunks):
        is_paragraph_end = chunk.endswith("\n\n")
        clean = chunk.strip()

        print(f"  Chunk {i+1}/{len(chunks)}: {clean[:70]}...")
        chunk_audio, sr = generate_single(clone_model, clean, voice_clone_prompt)
        audio_list.append(chunk_audio)
        paragraph_flags.append(is_paragraph_end)

        if on_chunk:
            on_chunk(i, len(chunks), chunk_audio, sr)

    result = stitch_chunks(audio_list, sr, paragraph_flags=paragraph_flags)
    del audio_list
    return result, sr


def generate_preview(
    text: str,
    clone_model,
    voice_clone_prompt,
) -> tuple[np.ndarray, int]:
    """Generate a short preview clip from the first sentence (max ~50 chars)."""
    # Take first sentence or first 50 chars, whichever is shorter
    preview_text = text.strip()
    for sep in (".", "!", "?"):
        idx = preview_text.find(sep)
        if 0 < idx < 120:
            preview_text = preview_text[: idx + 1]
            break
    if len(preview_text) > 120:
        preview_text = preview_text[:120].rsplit(" ", 1)[0] + "."

    return generate_single(clone_model, preview_text, voice_clone_prompt)


def generate_full(
    text: str,
    clone_model,
    voice_clone_prompt,
    on_chunk: Callable[[int, int, np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """prepare_text → single or chunked generation."""
    chunks = prepare_text(text)
    if len(chunks) == 1:
        print(f"Generating: {text[:80]}...")
        return generate_single(clone_model, chunks[0].strip(), voice_clone_prompt)
    print(f"Generating {len(chunks)} chunks...")
    return generate_chunked(chunks, clone_model, voice_clone_prompt, on_chunk=on_chunk)
