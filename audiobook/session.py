"""Crash recovery for audiobook production.

An audiobook takes hours to generate. The chunk audio and the chunk list (text, chapter,
audio path, approval) live in ~/.voicecloner/audiobook_session/ instead of a temp dir,
and the list is rewritten after every change, so a crash or a closed tab loses at most
the chunk being generated. "Resume last session" in either audiobook UI reloads it.

Both UIs keep chunks as the same five fields (the app's audiobook tab as dicts, the
standalone UI as ChunkState), so they share one session. Starting a new book (chunking
text) replaces it.
"""

from __future__ import annotations

import glob
import json
import os

SESSION_DIR = os.path.join(os.path.expanduser("~"), ".voicecloner", "audiobook_session")
_STATE = os.path.join(SESSION_DIR, "session.json")
FIELDS = ("text", "chapter", "is_para_end", "audio_path", "approved")


def chunk_path(name: str) -> str:
    """Where a chunk's audio goes, e.g. chunk_path("chunk_0007.wav")."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, name)


def start() -> None:
    """Begin a new book: drop the previous session's audio and chunk list."""
    for path in glob.glob(os.path.join(SESSION_DIR, "chunk_*.wav")):
        os.remove(path)
    if os.path.isfile(_STATE):
        os.remove(_STATE)


def save(chunks: list[dict], pronunciation: dict[str, str] | None = None) -> None:
    """Write the chunk list atomically, so a crash mid-write can't corrupt the last good one."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {"chunks": [{k: c.get(k) for k in FIELDS} for c in chunks]}
    if pronunciation is not None:
        data["pronunciation"] = pronunciation
    tmp = _STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _STATE)


def load() -> tuple[list[dict], dict[str, str] | None] | None:
    """Return (chunks, pronunciation) from the last session, or None if there is none.

    A chunk whose audio file has gone missing goes back to pending, so it gets regenerated
    instead of breaking the export.
    """
    if not os.path.isfile(_STATE):
        return None
    try:
        with open(_STATE, encoding="utf-8") as f:
            data = json.load(f)
        chunks = [{k: c.get(k) for k in FIELDS} for c in data["chunks"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    for c in chunks:
        c["text"] = c["text"] or ""
        c["is_para_end"] = bool(c["is_para_end"])
        if not (c["audio_path"] and os.path.isfile(c["audio_path"])):
            c["audio_path"], c["approved"] = None, False
        c["approved"] = bool(c["approved"])
    return chunks, data.get("pronunciation")


def summary(chunks: list[dict]) -> str:
    """One line for the UI: what the resumed session holds."""
    done = sum(1 for c in chunks if c["audio_path"])
    approved = sum(1 for c in chunks if c["approved"])
    chapters = len({c["chapter"] for c in chunks})
    return (f"Resumed {len(chunks)} chunks across {chapters} chapter(s): "
            f"{done} generated, {approved} approved.")
