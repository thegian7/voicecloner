"""
Qwen3-TTS Voice Cloner — Tom's setup

Usage:
    # Basic clone with your voice + style
    python clone.py "Whatever you want it to say"

    # Read from a text file
    python clone.py --file sherlock.txt -o sherlock.wav

    # Custom style description
    python clone.py "Hello world" --style "calm and thoughtful, like a narrator"

    # Custom output filename
    python clone.py "Hello world" -o myfile.wav

    # Use a different reference audio
    python clone.py "Hello world" --ref reference_calm.wav --ref-text "transcript of calm audio"

    # Skip VoiceDesign (just raw clone, faster)
    python clone.py "Hello world" --no-design
"""

import argparse
import re

import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# --- Defaults (edit these to taste) ---
DEFAULT_REF_AUDIO = "reference_excited.wav"
DEFAULT_REF_TEXT = "Hello! What up everybody, hope you are having an awesome day!"
DEFAULT_STYLE = (
    "A young excited American male with a warm energetic tone, "
    "speaking with enthusiasm and confidence like a podcast host"
)
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

CHUNK_TARGET_MIN = 100
CHUNK_TARGET_MAX = 200


def prepare_text(raw_text: str) -> list[str]:
    """Split raw text into generation-friendly chunks with pacing punctuation."""
    # Normalise whitespace
    text = raw_text.strip()

    # Split on paragraph breaks first, track boundaries
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for pi, para in enumerate(paragraphs):
        para = " ".join(para.split())  # collapse whitespace
        if not para:
            continue

        # Add breathing pauses: insert ... after commas in long clauses (>60 chars between punctuation)
        para = re.sub(r",(?=\s[^.!?]{60,})", ",...", para)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)

        # Group short sentences together into chunks of ~100-200 chars
        current = ""
        for sent in sentences:
            if not sent:
                continue
            if current and len(current) + len(sent) + 1 > CHUNK_TARGET_MAX:
                chunks.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent

        if current:
            chunks.append(current.strip())

        # Mark paragraph boundary (except after last paragraph)
        if pi < len(paragraphs) - 1 and chunks:
            chunks[-1] = chunks[-1] + "\n\n"  # sentinel for paragraph pause

    return chunks


def crossfade(a: np.ndarray, b: np.ndarray, overlap_samples: int) -> np.ndarray:
    """Crossfade two audio arrays over overlap_samples."""
    if overlap_samples <= 0 or len(a) < overlap_samples or len(b) < overlap_samples:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=a.dtype)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=b.dtype)
    blended = a[-overlap_samples:] * fade_out + b[:overlap_samples] * fade_in
    return np.concatenate([a[:-overlap_samples], blended, b[overlap_samples:]])


def generate_chunked(
    chunks: list[str],
    clone_model,
    voice_clone_prompt,
    pause_ms: int = 250,
    paragraph_pause_ms: int = 500,
    crossfade_ms: int = 40,
) -> tuple:
    """Generate audio per chunk and stitch with crossfades and silence."""
    result = None
    sr = None

    for i, chunk in enumerate(chunks):
        is_paragraph_end = chunk.endswith("\n\n")
        clean_chunk = chunk.strip()

        print(f"  Chunk {i+1}/{len(chunks)}: {clean_chunk[:70]}...")
        wavs, sr = clone_model.generate_voice_clone(
            text=clean_chunk,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
        )
        chunk_audio = wavs[0]

        if result is None:
            result = chunk_audio
        else:
            # Add silence gap then crossfade into next chunk
            gap = paragraph_pause_ms if is_paragraph_end else pause_ms
            silence = np.zeros(int(sr * gap / 1000), dtype=result.dtype)
            result = np.concatenate([result, silence])
            overlap = int(sr * crossfade_ms / 1000)
            result = crossfade(result, chunk_audio, overlap)

    return result, sr


def generate(text, output, ref_audio, ref_text, style, use_design):
    if use_design:
        print("Loading VoiceDesign model...")
        design_model = Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL, device_map="mps", dtype=torch.float16
        )
        print(f"Generating voice style: {style[:60]}...")
        design_model.generate_voice_design(
            text="Hello everybody, hope you are having an awesome day!",
            language="English",
            instruct=style,
        )
        del design_model
        torch.mps.empty_cache()

    print("Loading Clone model...")
    clone_model = Qwen3TTSModel.from_pretrained(
        CLONE_MODEL, device_map="mps", dtype=torch.float32
    )

    print("Building voice clone prompt...")
    voice_clone_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    chunks = prepare_text(text)
    if len(chunks) == 1:
        print(f"Generating: {text[:80]}...")
        wavs, sr = clone_model.generate_voice_clone(
            text=chunks[0].strip(),
            language="English",
            voice_clone_prompt=voice_clone_prompt,
        )
        audio, sr = wavs[0], sr
    else:
        print(f"Generating {len(chunks)} chunks...")
        audio, sr = generate_chunked(chunks, clone_model, voice_clone_prompt)

    sf.write(output, audio, sr)
    print(f"Saved {output} ({len(audio)/sr:.1f}s, {sr}Hz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TomClone TTS")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--file", "-f", help="Read text from a file instead")
    parser.add_argument("-o", "--output", default="output.wav", help="Output filename")
    parser.add_argument("--ref", default=DEFAULT_REF_AUDIO, help="Reference audio file")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Transcript of reference audio")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="VoiceDesign style description")
    parser.add_argument("--no-design", action="store_true", help="Skip VoiceDesign, raw clone only")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text as an argument or use --file")

    generate(text, args.output, args.ref, args.ref_text, args.style, not args.no_design)
