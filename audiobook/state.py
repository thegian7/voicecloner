"""Shared state types for the audiobook UI."""

from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass, field

from core import DEFAULT_STYLE  # noqa: F401  (re-exported for the UI)
from . import session


# Previews and exports. Chunk audio lives in the session dir (see session.py) so it
# survives a crash.
TEMP_DIR = tempfile.mkdtemp(prefix="audiobook_")

# Common TTS-friendly replacements
DEFAULT_PRONUNCIATION: dict[str, str] = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Dr.": "Doctor",
    "St.": "Saint",
}


@dataclass
class ChunkState:
    """One text chunk and its generation state."""

    text: str
    chapter: str
    is_para_end: bool = False
    audio_path: str | None = None
    approved: bool = False


@dataclass
class BookState:
    """Full audiobook session state — serialised into gr.State."""

    chunks: list[ChunkState] = field(default_factory=list)
    pronunciation: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PRONUNCIATION))

    # --- pronunciation helpers ---

    def apply_pronunciation(self, text: str) -> str:
        """Apply pronunciation dictionary to text."""
        result = text
        for find, replace in self.pronunciation.items():
            result = re.sub(re.escape(find), replace, result)
        return result

    def update_pronunciation(self, raw: str) -> None:
        """Parse 'find -> replace' lines into the pronunciation dict."""
        self.pronunciation = dict(DEFAULT_PRONUNCIATION)
        for line in raw.strip().splitlines():
            if "->" in line:
                parts = line.split("->", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                if key and val:
                    self.pronunciation[key] = val

    def pronunciation_as_text(self) -> str:
        """Serialise pronunciation dict to editable text."""
        return "\n".join(f"{k} -> {v}" for k, v in self.pronunciation.items())

    # --- chunk helpers ---

    @property
    def chapter_names(self) -> list[str]:
        """Ordered unique chapter names."""
        seen: set[str] = set()
        names: list[str] = []
        for c in self.chunks:
            if c.chapter not in seen:
                seen.add(c.chapter)
                names.append(c.chapter)
        return names

    def chunks_for_chapter(self, chapter: str) -> list[tuple[int, ChunkState]]:
        """Return (global_index, chunk) pairs for a chapter."""
        return [(i, c) for i, c in enumerate(self.chunks) if c.chapter == chapter]

    # --- crash recovery (see session.py) ---

    def save(self) -> None:
        """Persist the chunk list so the session can be resumed after a crash."""
        session.save([asdict(c) for c in self.chunks], self.pronunciation)

    @classmethod
    def resume(cls) -> BookState | None:
        """The last saved session, or None if there isn't one."""
        saved = session.load()
        if saved is None:
            return None
        chunks, pronunciation = saved
        return cls(
            chunks=[ChunkState(**c) for c in chunks],
            pronunciation=pronunciation or dict(DEFAULT_PRONUNCIATION),
        )

    def progress_summary(self) -> dict[str, dict[str, int]]:
        """Per-chapter counts: pending / generated / approved."""
        summary: dict[str, dict[str, int]] = {}
        for c in self.chunks:
            if c.chapter not in summary:
                summary[c.chapter] = {"pending": 0, "generated": 0, "approved": 0}
            bucket = summary[c.chapter]
            if c.approved:
                bucket["approved"] += 1
            elif c.audio_path:
                bucket["generated"] += 1
            else:
                bucket["pending"] += 1
        return summary
