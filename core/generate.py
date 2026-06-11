from typing import Callable, Iterator

import numpy as np

from .text import prepare_text
from .audio import stitch_chunks


# Generation defaults. VoxCPM2 docs recommend cfg_value=2.0 and 7-10 timesteps.
DEFAULT_CFG = 2.0
DEFAULT_TIMESTEPS = 10


def run_voice_design(model, style: str) -> None:
    """No-op kept for backward compatibility.

    In the Qwen3-TTS architecture this ran a separate VoiceDesign model to
    condition the clone on a style. VoxCPM2 bakes the style directly into the
    text prompt at generation time (see `_styled_text`), so this is a no-op now.
    Callers can keep invoking it; pass `style=` to `create_clone_prompt` instead.
    """
    return None


def create_clone_prompt(model, ref_audio: str, ref_text: str = "", style: str = "") -> dict:
    """Build a voice prompt for VoxCPM2.

    Returns a plain dict rather than a model-specific prompt object — VoxCPM2
    doesn't pre-compute prompts; it just needs the paths/strings at generate time.
    """
    return {
        "ref_audio": ref_audio,
        "ref_text": ref_text or "",
        "style": (style or "").strip(),
    }


def _styled_text(text: str, style: str) -> str:
    """VoxCPM2 takes style cues as a parenthesized prefix on the text."""
    s = (style or "").strip()
    return f"({s}){text}" if s else text


def _sample_rate(model) -> int:
    # VoxCPM2 exposes the output rate on the inner tts_model (48000 for v2).
    return int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000))


def generate_single(model, text: str, voice_prompt: dict) -> tuple[np.ndarray, int]:
    """Generate audio for a single text chunk."""
    ref_audio = voice_prompt.get("ref_audio")
    ref_text = voice_prompt.get("ref_text", "")
    # VoxCPM2's parens-prefix style works in voice-design mode (no ref audio).
    # With a reference audio the model often reads the description aloud instead
    # of using it as a cue — so we drop it. The reference IS the voice character.
    styled = text if ref_audio else _styled_text(text, voice_prompt.get("style", ""))

    audio = model.generate(
        text=styled,
        prompt_wav_path=ref_audio,
        prompt_text=ref_text,
        reference_wav_path=ref_audio,
        cfg_value=DEFAULT_CFG,
        inference_timesteps=DEFAULT_TIMESTEPS,
    )
    return np.asarray(audio, dtype=np.float32), _sample_rate(model)


def generate_chunked(
    chunks: list[str],
    clone_model,
    voice_clone_prompt: dict,
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
    voice_clone_prompt: dict,
) -> tuple[np.ndarray, int]:
    """Generate a short preview clip from the first sentence (max ~120 chars)."""
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
    voice_clone_prompt: dict,
    on_chunk: Callable[[int, int, np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """prepare_text → single or chunked generation."""
    chunks = prepare_text(text)
    if len(chunks) == 1:
        print(f"Generating: {text[:80]}...")
        return generate_single(clone_model, chunks[0].strip(), voice_clone_prompt)
    print(f"Generating {len(chunks)} chunks...")
    return generate_chunked(chunks, clone_model, voice_clone_prompt, on_chunk=on_chunk)


def generate_voice_design(
    model,
    description: str,
    sample_text: str,
) -> tuple[np.ndarray, int]:
    """Generate a voice sample from a natural-language description, no reference audio.

    VoxCPM2's voice-design mode: the parens-prefix describes a voice and the
    model invents one matching the description. Useful for bootstrapping
    character voices when no recording is available. Save the returned audio as
    a reference voice for subsequent cloning to get consistent output.
    """
    if not description.strip():
        raise ValueError("description must be non-empty for voice-design generation")
    styled = f"({description.strip()}){sample_text}"
    audio = model.generate(
        text=styled,
        cfg_value=DEFAULT_CFG,
        inference_timesteps=DEFAULT_TIMESTEPS,
    )
    return np.asarray(audio, dtype=np.float32), _sample_rate(model)


def generate_stream(
    text: str,
    model,
    voice_prompt: dict,
) -> Iterator[tuple[np.ndarray, int]]:
    """Stream audio as it's generated. Yields (chunk_audio, sample_rate) tuples.

    Splits long text via prepare_text, then for each text chunk uses
    VoxCPM2's generate_streaming to emit audio sub-chunks as they're produced.
    """
    ref_audio = voice_prompt.get("ref_audio")
    ref_text = voice_prompt.get("ref_text", "")
    sr = _sample_rate(model)

    text_chunks = prepare_text(text)
    for i, text_chunk in enumerate(text_chunks):
        clean = text_chunk.strip()
        if not clean:
            continue
        # Style is not applied in cloning mode (see generate_single).
        styled = clean if ref_audio else _styled_text(clean, voice_prompt.get("style", ""))
        print(f"  Streaming chunk {i+1}/{len(text_chunks)}: {clean[:70]}...")

        for audio in model.generate_streaming(
            text=styled,
            prompt_wav_path=ref_audio,
            prompt_text=ref_text,
            reference_wav_path=ref_audio,
            cfg_value=DEFAULT_CFG,
            inference_timesteps=DEFAULT_TIMESTEPS,
        ):
            yield np.asarray(audio, dtype=np.float32), sr
