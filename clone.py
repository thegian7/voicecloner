"""
VoxCPM2 Voice Cloner — CLI

Usage:
    python clone.py "Whatever you want it to say"
    python clone.py --file sherlock.txt -o sherlock.wav
    python clone.py "Hello world" --style "calm narrator" --no-design
    python clone.py "Hello world" --ref reference.wav --ref-text "transcript"
"""

import argparse

from core import (
    load_tts_model,
    create_clone_prompt,
    generate_full,
    save_wav,
)

DEFAULT_REF_AUDIO = "reference_excited.wav"
DEFAULT_REF_TEXT = "Hello! What up everybody, hope you are having an awesome day!"
DEFAULT_STYLE = (
    "A young excited American male with a warm energetic tone, "
    "speaking with enthusiasm and confidence like a podcast host"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceCloner TTS CLI")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--file", "-f", help="Read text from a file instead")
    parser.add_argument("-o", "--output", default="output.wav", help="Output filename")
    parser.add_argument("--ref", default=DEFAULT_REF_AUDIO, help="Reference audio file")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Transcript of reference audio")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="Voice style description (prepended to prompt)")
    parser.add_argument("--no-design", action="store_true", help="Skip style prompt, raw clone only")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text as an argument or use --file")

    style = "" if args.no_design else args.style

    model = load_tts_model()
    voice_clone_prompt = create_clone_prompt(model, args.ref, args.ref_text, style=style)
    audio, sr = generate_full(text, model, voice_clone_prompt)
    save_wav(args.output, audio, sr)
    print(f"Saved {args.output} ({len(audio)/sr:.1f}s, {sr}Hz)")
