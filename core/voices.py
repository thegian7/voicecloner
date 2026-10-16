"""
Voice library — save, load, and manage cloned voice profiles.
Stores voice profiles in ~/.voicecloner/voices/ with reference audio + metadata.

A profile can also carry a LoRA fine-tuned on that speaker (voices/<name>/lora/,
holding lora_weights.safetensors + lora_config.json from VoxCPM's trainer). The
LoRA travels with the profile's reference recording: any voice prompt built from
that recording picks it up (see lora_for_ref), so every entry point — the Gradio
app, the TTS server, the audiobook pipeline, the CLI — uses it without changes.
"""

import hashlib
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
    # Not stored in meta.json (older builds load meta.json straight into this class and
    # would reject the key); filled in on load from voices/<name>/lora/ if it exists.
    lora_path: str = ""


def _ensure_dir() -> None:
    os.makedirs(VOICES_DIR, exist_ok=True)


def _voice_dir(name: str) -> str:
    return os.path.join(VOICES_DIR, name)


def _meta_path(name: str) -> str:
    return os.path.join(_voice_dir(name), "meta.json")


def _lora_dir(name: str) -> str:
    return os.path.join(_voice_dir(name), "lora")


def _from_meta(data: dict) -> VoiceProfile:
    data = {k: v for k, v in data.items() if k != "lora_path"}
    profile = VoiceProfile(**data)
    if os.path.isfile(os.path.join(_lora_dir(profile.name), "lora_weights.safetensors")):
        profile.lora_path = _lora_dir(profile.name)
    return profile


def _write_meta(profile: VoiceProfile) -> None:
    data = {k: v for k, v in asdict(profile).items() if k != "lora_path"}
    with open(_meta_path(profile.name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    _write_meta(profile)
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
            profiles.append(_from_meta(data))
    profiles.sort(key=lambda p: p.created_at, reverse=True)
    return profiles


def load_voice(name: str) -> VoiceProfile:
    """Load a single voice profile by name."""
    meta = _meta_path(name)
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"Voice '{name}' not found.")
    with open(meta, encoding="utf-8") as f:
        return _from_meta(json.load(f))


def attach_lora(name: str, lora_dir: str) -> VoiceProfile:
    """Copy a trained LoRA checkpoint into a saved voice and record it in meta.json."""
    for f in ("lora_weights.safetensors", "lora_config.json"):
        if not os.path.isfile(os.path.join(lora_dir, f)):
            raise FileNotFoundError(f"{lora_dir} has no {f} — expected a VoxCPM LoRA checkpoint dir.")
    profile = load_voice(name)
    dest = _lora_dir(name)
    os.makedirs(dest, exist_ok=True)
    for f in ("lora_weights.safetensors", "lora_config.json"):
        shutil.copy2(os.path.join(lora_dir, f), os.path.join(dest, f))
    profile.lora_path = dest
    _ref_hashes.clear()
    return profile


_ref_hashes: dict[str, tuple[int, str, str]] = {}  # ref path -> (size, sha1, lora_path)


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lora_for_ref(ref_audio: str | None) -> str:
    """Return the LoRA dir of the saved voice whose reference recording this is, or "".

    Matched by content, not path: the Gradio app hands generation a temp copy of
    the reference, so a path match would miss it.
    """
    if not ref_audio or not os.path.isfile(ref_audio) or not os.path.isdir(VOICES_DIR):
        return ""
    if not _ref_hashes:
        for v in list_voices():
            if v.lora_path and os.path.isdir(v.lora_path) and os.path.isfile(v.ref_audio_path):
                _ref_hashes[v.ref_audio_path] = (os.path.getsize(v.ref_audio_path),
                                                 _sha1(v.ref_audio_path), v.lora_path)
    size = os.path.getsize(ref_audio)
    candidates = [(h, lp) for s, h, lp in _ref_hashes.values() if s == size]
    if not candidates:
        return ""
    digest = _sha1(ref_audio)
    return next((lp for h, lp in candidates if h == digest), "")


def delete_voice(name: str) -> None:
    """Delete a voice profile and its files from disk."""
    vdir = _voice_dir(name)
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
