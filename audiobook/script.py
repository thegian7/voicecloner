"""Tagged-script format parser for cast-aware audiobook generation.

Format:
    # Chapter 2: The Sound of Thinking

    NARRATOR: The fog came in at 4:47 AM, thirteen minutes before Maren's alarm.
    She knew this because she was already awake.

    MAREN: CORA, are you there?

    CORA: I'm here. You're home late.

    ---

    NARRATOR: Next section.

Rules:
- `# Chapter N: title` sets the current chapter context.
- `SPEAKER: text` starts a script line; SPEAKER is all-caps (letters, digits, underscores, spaces).
- Lines that don't match SPEAKER: are continuation of the previous block (re-joined with a single space).
- Blank line ends a block.
- `---` is a visual separator; ignored.
- `//` or `#` at start of line (not chapter heading) is a comment; ignored.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9_]*(?:\s+[A-Z][A-Z0-9_]+)*)\s*:\s*(.*)$")
_CHAPTER_RE = re.compile(r"^#\s+(.*)$")


@dataclass
class ScriptLine:
    chapter: str
    speaker: str
    text: str
    line_no: int  # 1-based source line where the block started

    @property
    def slug(self) -> str:
        """Short identifier for filenames."""
        return f"{self.line_no:05d}_{self.speaker.replace(' ', '_').lower()}"


def parse_script(path: str) -> list[ScriptLine]:
    """Parse one tagged script file. Returns list of ScriptLine in source order."""
    lines: list[ScriptLine] = []
    chapter = "Unknown Chapter"
    current_speaker: str | None = None
    current_parts: list[str] = []
    current_line_no = 0

    def flush() -> None:
        nonlocal current_speaker, current_parts, current_line_no
        if current_speaker and current_parts:
            text = " ".join(p.strip() for p in current_parts if p.strip()).strip()
            if text:
                lines.append(
                    ScriptLine(
                        chapter=chapter,
                        speaker=current_speaker,
                        text=text,
                        line_no=current_line_no,
                    )
                )
        current_speaker = None
        current_parts = []
        current_line_no = 0

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                flush()
                continue

            if stripped == "---":
                flush()
                continue

            if stripped.startswith("//"):
                continue

            ch = _CHAPTER_RE.match(stripped)
            if ch:
                flush()
                chapter = ch.group(1).strip()
                continue

            sp = _SPEAKER_RE.match(stripped)
            if sp:
                flush()
                current_speaker = sp.group(1).strip()
                tail = sp.group(2).strip()
                current_parts = [tail] if tail else []
                current_line_no = lineno
                continue

            # Continuation of previous block
            if current_speaker:
                current_parts.append(stripped)
            # else: stray text before any speaker tag — silently ignore

    flush()
    return lines


def parse_script_path(path: str) -> list[ScriptLine]:
    """Parse a file, or every .md file in a directory (sorted by name)."""
    if os.path.isdir(path):
        out: list[ScriptLine] = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".md") and not name.startswith("_"):
                out.extend(parse_script(os.path.join(path, name)))
        return out
    return parse_script(path)


def speakers_in(lines: list[ScriptLine]) -> list[str]:
    """Return unique speaker tags in order of first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln.speaker not in seen:
            seen.add(ln.speaker)
            out.append(ln.speaker)
    return out


def chapters_in(lines: list[ScriptLine]) -> list[str]:
    """Return unique chapter names in order of first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln.chapter not in seen:
            seen.add(ln.chapter)
            out.append(ln.chapter)
    return out
