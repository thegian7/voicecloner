"""Cast manifest: speaker tag → voice library entry.

Format (TOML):
    default = "NARRATOR"

    [cast]
    NARRATOR = "lattice_narrator"
    MAREN    = "lattice_maren"
    CORA     = "lattice_cora"
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field

from core import VoiceProfile, list_voices, load_voice


@dataclass
class Cast:
    default: str
    speakers: dict[str, str] = field(default_factory=dict)
    _prompts: dict[str, dict] = field(default_factory=dict)

    def voice_name_for(self, speaker: str) -> str:
        """Resolve a speaker tag to a voice library name. Falls back to default."""
        if speaker in self.speakers:
            return self.speakers[speaker]
        fallback = self.speakers.get(self.default)
        if fallback is None:
            raise KeyError(
                f"Speaker '{speaker}' not in cast and default '{self.default}' "
                "has no voice mapping."
            )
        return fallback

    def voice_profile(self, speaker: str) -> VoiceProfile:
        return load_voice(self.voice_name_for(speaker))


def load_cast(path: str) -> Cast:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    default = data.get("default")
    speakers = data.get("cast", {})

    if not default:
        raise ValueError(f"{path}: missing top-level `default` field")
    if not speakers:
        raise ValueError(f"{path}: empty `[cast]` table")
    if default not in speakers:
        raise ValueError(
            f"{path}: default speaker '{default}' is not present in [cast]"
        )

    return Cast(default=default, speakers=dict(speakers))


def validate_cast_against_voices(cast: Cast) -> tuple[list[str], list[str]]:
    """Check which voice library entries referenced by the cast exist on disk.

    Returns (present, missing) lists of voice library names.
    """
    available = {v.name for v in list_voices()}
    present: list[str] = []
    missing: list[str] = []
    for speaker, voice in cast.speakers.items():
        if voice in available:
            present.append(voice)
        else:
            missing.append(voice)
    return present, missing


def validate_cast_against_script(cast: Cast, speakers_seen: list[str]) -> list[str]:
    """Return script speakers that aren't in the cast (will fall back to default)."""
    return [s for s in speakers_seen if s not in cast.speakers]


def main():
    """CLI: `python -m audiobook.cast <cast.toml>` — print cast + validation report."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a cast manifest")
    parser.add_argument("cast", help="Path to cast.toml")
    parser.add_argument("--script", help="Optional script file/dir to validate against")
    args = parser.parse_args()

    cast = load_cast(args.cast)
    print(f"Default speaker: {cast.default}")
    print(f"Cast ({len(cast.speakers)} speakers):")
    for spk, voice in cast.speakers.items():
        print(f"  {spk:24s} -> {voice}")

    present, missing = validate_cast_against_voices(cast)
    if missing:
        print(f"\nMissing voices (need bootstrap): {missing}")
    else:
        print(f"\nAll {len(present)} cast voices present in library.")

    if args.script:
        from .script import parse_script_path, speakers_in
        lines = parse_script_path(args.script)
        seen = speakers_in(lines)
        unmapped = validate_cast_against_script(cast, seen)
        print(f"\nScript speakers: {seen}")
        if unmapped:
            print(f"Speakers without cast mapping (will fall back to '{cast.default}'):")
            for spk in unmapped:
                print(f"  - {spk}")
        else:
            print("All script speakers are in cast.")


if __name__ == "__main__":
    main()
