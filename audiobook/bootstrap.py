"""Bootstrap character voices via VoxCPM2 voice-design.

Reads a voices.toml describing each character with a natural-language voice
description + a sample text. For each character not already in the local voice
library, generates a sample via VoxCPM2's voice-design mode and saves it as a
reference voice for subsequent cloning. This is the cheap path to a multi-voice
cast: no recording session required, just text descriptions.

voices.toml format:

    [voices.lattice_narrator]
    description = '''
    Female literary fiction narrator, mid-pitch, contemplative, unhurried,
    American English. Reading register, not performative.
    '''
    sample_text = "The fog came in at 4:47 AM, thirteen minutes before her alarm."

    [voices.lattice_maren]
    description = '''
    American woman, ~35, San Francisco. Mid-pitch, dry, technically precise.
    Engineer shorthand. Self-correcting mid-sentence.
    '''
    sample_text = "I know what my cortisol levels suggest."

CLI:
    python -m audiobook.bootstrap --voices voices.toml [--only lattice_maren] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from dataclasses import dataclass

# Add parent dir to path so `from core import ...` works when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ModelCache,
    generate_voice_design,
    list_voices,
    save_voice,
    save_wav,
)


@dataclass
class VoiceSpec:
    name: str
    description: str
    sample_text: str


def parse_voices_toml(path: str) -> list[VoiceSpec]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    entries = data.get("voices", {})
    if not entries:
        raise ValueError(f"{path}: no [voices.*] entries found")

    specs: list[VoiceSpec] = []
    for name, body in entries.items():
        desc = (body.get("description") or "").strip()
        sample = (body.get("sample_text") or "").strip()
        if not desc:
            raise ValueError(f"voice '{name}': missing description")
        if not sample:
            raise ValueError(f"voice '{name}': missing sample_text")
        specs.append(VoiceSpec(name=name, description=desc, sample_text=sample))
    return specs


def bootstrap_one(spec: VoiceSpec, cache: ModelCache, work_dir: str) -> str:
    """Generate the sample voice and save it to the library. Returns the saved path."""
    print(f"\n=== {spec.name} ===")
    print(f"  description: {spec.description[:120]}...")
    print(f"  sample_text: {spec.sample_text}")

    audio, sr = generate_voice_design(cache.tts, spec.description, spec.sample_text)
    os.makedirs(work_dir, exist_ok=True)
    sample_path = os.path.join(work_dir, f"{spec.name}.wav")
    save_wav(sample_path, audio, sr)
    print(f"  wrote {sample_path} ({len(audio)/sr:.1f}s @ {sr}Hz)")

    profile = save_voice(
        name=spec.name,
        ref_audio=sample_path,
        ref_text=spec.sample_text,
        style="",  # style not used in cloning mode; description was bootstrap-only
        use_design=False,
    )
    print(f"  saved voice library entry: {profile.ref_audio_path}")
    return profile.ref_audio_path


def main():
    parser = argparse.ArgumentParser(description="Bootstrap character voices via VoxCPM2 voice-design")
    parser.add_argument("--voices", required=True, help="Path to voices.toml")
    parser.add_argument("--only", action="append", help="Only bootstrap these names (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="Re-bootstrap voices already in library")
    parser.add_argument(
        "--work-dir",
        default=os.path.expanduser("~/.voicecloner/bootstrap"),
        help="Where to stage the generated WAVs before saving to the library",
    )
    args = parser.parse_args()

    specs = parse_voices_toml(args.voices)
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"No voices in {args.voices} match --only {args.only}")
            sys.exit(1)

    existing = {v.name for v in list_voices()}
    to_make: list[VoiceSpec] = []
    skipped: list[str] = []
    for spec in specs:
        if spec.name in existing and not args.overwrite:
            skipped.append(spec.name)
        else:
            to_make.append(spec)

    if skipped:
        print(f"Skipping (already in library, use --overwrite to redo): {skipped}")
    if not to_make:
        print("Nothing to do.")
        return

    print(f"Bootstrapping {len(to_make)} voice(s)...")
    cache = ModelCache()
    _ = cache.tts  # force load now so the first voice doesn't pay the load cost alone

    for spec in to_make:
        try:
            bootstrap_one(spec, cache, args.work_dir)
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone. Saved voices:")
    for v in list_voices():
        print(f"  - {v.name}")


if __name__ == "__main__":
    main()
