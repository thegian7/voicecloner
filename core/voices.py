"""
Voice library — save, load, and manage cloned voice profiles.
Stores voice profiles in ~/.voicecloner/voices/ with reference audio + metadata.
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


VOICES_DIR = os.path.join(os.path.expanduser("~"), ".voicecloner", "voices")


@dataclass
class VoiceProfile:
    name: str
    ref_audio_path: str
    ref_text: str
    style: str
    use_design: bool
    created_at: str


def _ensure_dir() -> None:
    os.makedirs(VOICES_DIR, exist_ok=True)


def _voice_dir(name: str) -> str:
    return os.path.join(VOICES_DIR, name)


def _meta_path(name: str) -> str:
    return os.path.join(_voice_dir(name), "meta.json")


def save_voice(
    name: str,
    ref_audio: str,
    ref_text: str,
    style: str,
    use_design: bool,
) -> VoiceProfile:
    """Save a voice profile to disk, copying the reference audio."""
    _ensure_dir()
    vdir = _voice_dir(name)
    os.makedirs(vdir, exist_ok=True)

    ext = os.path.splitext(ref_audio)[1] or ".wav"
    dest_audio = os.path.join(vdir, f"ref{ext}")
    shutil.copy2(ref_audio, dest_audio)

    profile = VoiceProfile(
        name=name,
        ref_audio_path=dest_audio,
        ref_text=ref_text,
        style=style,
        use_design=use_design,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(_meta_path(name), "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2)

    return profile


def list_voices() -> list[VoiceProfile]:
    """Return all saved voice profiles sorted by creation date (newest first)."""
    _ensure_dir()
    profiles: list[VoiceProfile] = []
    for entry in os.listdir(VOICES_DIR):
        meta = os.path.join(VOICES_DIR, entry, "meta.json")
        if os.path.isfile(meta):
            with open(meta, encoding="utf-8") as f:
                data = json.load(f)
            profiles.append(VoiceProfile(**data))
    profiles.sort(key=lambda p: p.created_at, reverse=True)
    return profiles


def load_voice(name: str) -> VoiceProfile:
    """Load a single voice profile by name."""
    meta = _meta_path(name)
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"Voice '{name}' not found.")
    with open(meta, encoding="utf-8") as f:
        return VoiceProfile(**json.load(f))


def delete_voice(name: str) -> None:
    """Delete a voice profile and its files from disk."""
    vdir = _voice_dir(name)
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
