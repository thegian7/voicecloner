"""
Text preprocessor for audiobook production.
Handles EPUB import, chapter detection, abbreviation/number expansion,
and quote normalization to produce clean text for TTS.
"""

import os
import re
from dataclasses import dataclass

import inflect

_inflect = inflect.engine()


@dataclass
class Chapter:
    title: str
    text: str


# ── Quote / punctuation normalization ────────────────────────────────

def normalize_quotes(text: str) -> str:
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # curly double
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # curly single
    text = text.replace("\u2014", ", ")                          # em-dash → comma pause
    text = text.replace("\u2013", "-")                           # en-dash → hyphen
    text = text.replace("\u2026", "...")                         # ellipsis char
    return text


# ── Abbreviation expansion ───────────────────────────────────────────

_ABBREVIATIONS: dict[str, str] = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    # St. handled separately via context heuristic
    "Jr.": "Junior",
    "Sr.": "Senior",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Gen.": "General",
    "Gov.": "Governor",
    "Sgt.": "Sergeant",
    "Capt.": "Captain",
    "Lt.": "Lieutenant",
    "Col.": "Colonel",
    "vs.": "versus",
    "etc.": "etcetera",
    "approx.": "approximately",
    "dept.": "department",
    "est.": "established",
    "govt.": "government",
}

# Build pattern: match whole-word abbreviations (case-sensitive)
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")",
)


_ST_STREET_RE = re.compile(
    r"(?:"
    r"(?:\d+(?:st|nd|rd|th)?)\s+St\."       # "42nd St." / "5 St."
    r"|(?:[\w]+\s+)St\.\s*(?:$|[,\d])"      # "Elm St." before comma/EOL/number
    r")",
    re.IGNORECASE,
)


def _expand_st(text: str) -> str:
    """Context-aware St. expansion: Street after numbers/road names, Saint otherwise."""
    # First pass: mark street contexts
    def _mark_street(m: re.Match) -> str:
        return m.group(0).replace("St.", "Street")

    text = _ST_STREET_RE.sub(_mark_street, text)
    # Remaining St. → Saint
    text = re.sub(r"\bSt\.", "Saint", text)
    return text


def expand_abbreviations(text: str) -> str:
    text = _expand_st(text)
    return _ABBREV_PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(0)], text)


# ── Number expansion ────────────────────────────────────────────────

_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DOLLAR_RE = re.compile(r"\$([\d,]+)(?:\.(\d{2}))?")
_PLAIN_NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.ordinal(_inflect.number_to_words(int(m.group(1))))


def _expand_dollar(m: re.Match) -> str:
    dollars = int(m.group(1).replace(",", ""))
    cents = int(m.group(2)) if m.group(2) else 0
    parts = []
    if dollars:
        parts.append(f"{_inflect.number_to_words(dollars)} dollar{'s' if dollars != 1 else ''}")
    if cents:
        parts.append(f"{_inflect.number_to_words(cents)} cent{'s' if cents != 1 else ''}")
    return " and ".join(parts) if parts else "zero dollars"


def _expand_plain_number(m: re.Match) -> str:
    num_str = m.group(0).replace(",", "")
    try:
        if "." in num_str:
            return _inflect.number_to_words(float(num_str))
        return _inflect.number_to_words(int(num_str))
    except (ValueError, OverflowError):
        return m.group(0)


def expand_numbers(text: str) -> str:
    text = _ORDINAL_RE.sub(_expand_ordinal, text)
    text = _DOLLAR_RE.sub(_expand_dollar, text)
    text = _PLAIN_NUM_RE.sub(_expand_plain_number, text)
    return text


# ── Header / metadata stripping ─────────────────────────────────────

_PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$")
_RUNNING_HEADER_RE = re.compile(
    r"^\s*(chapter\s+\w+|part\s+\w+)\s*$", re.IGNORECASE
)


def strip_headers_metadata(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if _PAGE_NUM_RE.match(line):
            continue
        if _RUNNING_HEADER_RE.match(line):
            # Keep as a marker for chapter detection but don't speak it
            cleaned.append(f"\n[CHAPTER_MARKER]{line.strip()}[/CHAPTER_MARKER]\n")
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ── Chapter detection ────────────────────────────────────────────────

_CHAPTER_PATTERNS = [
    re.compile(
        r"^\s*\[CHAPTER_MARKER\](.*?)\[/CHAPTER_MARKER\]\s*$", re.MULTILINE
    ),
    re.compile(
        r"^\s*(chapter\s+\w+[\s:—\-]*.*)$", re.IGNORECASE | re.MULTILINE
    ),
    re.compile(
        r"^\s*(part\s+[IVXLCDM\d]+[\s:—\-]*.*)$", re.IGNORECASE | re.MULTILINE
    ),
    re.compile(r"^\s*(\*\s*\*\s*\*)\s*$", re.MULTILINE),
]


def detect_chapters(text: str) -> list[Chapter]:
    # Find all split points
    splits: list[tuple[int, str]] = []
    for pat in _CHAPTER_PATTERNS:
        for m in pat.finditer(text):
            title = m.group(1).strip()
            # Clean marker tags
            title = title.replace("[CHAPTER_MARKER]", "").replace("[/CHAPTER_MARKER]", "")
            splits.append((m.start(), title))

    if not splits:
        # No chapters detected — return as single chapter
        clean = re.sub(
            r"\[CHAPTER_MARKER\].*?\[/CHAPTER_MARKER\]", "", text
        ).strip()
        return [Chapter(title="Full Text", text=clean)] if clean else []

    # Sort by position, deduplicate overlapping
    splits.sort(key=lambda x: x[0])

    chapters: list[Chapter] = []
    for i, (pos, title) in enumerate(splits):
        end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        body = text[pos:end]
        # Remove the heading line itself and markers from body
        body = re.sub(
            r"\[CHAPTER_MARKER\].*?\[/CHAPTER_MARKER\]", "", body
        )
        # Remove the chapter heading line from body text
        for pat in _CHAPTER_PATTERNS:
            body = pat.sub("", body, count=1)
        body = body.strip()
        if not body:
            continue
        if title == "* * *":
            title = f"Scene Break {len(chapters) + 1}"
        chapters.append(Chapter(title=title, text=body))

    # If text exists before first split, prepend it
    if splits[0][0] > 0:
        preamble = text[: splits[0][0]]
        preamble = re.sub(
            r"\[CHAPTER_MARKER\].*?\[/CHAPTER_MARKER\]", "", preamble
        ).strip()
        if preamble:
            chapters.insert(0, Chapter(title="Preamble", text=preamble))

    return chapters


# ── EPUB extraction ──────────────────────────────────────────────────

def extract_epub(path: str) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path, options={"ignore_ncx": True})
    texts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        body = soup.find("body")
        if not body:
            continue
        # Convert <p> and <br> to newlines, extract text
        for br in body.find_all("br"):
            br.replace_with("\n")
        for p in body.find_all("p"):
            p.insert_after("\n\n")
        texts.append(body.get_text())

    return "\n\n".join(texts)


# ── Main entry points ────────────────────────────────────────────────

def preprocess(raw_text: str) -> list[Chapter]:
    text = normalize_quotes(raw_text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = strip_headers_metadata(text)
    chapters = detect_chapters(text)
    return chapters


def preprocess_epub(path: str) -> list[Chapter]:
    raw = extract_epub(path)
    return preprocess(raw)


def preprocess_file(path: str) -> list[Chapter]:
    """Load an EPUB or UTF-8 text file. Unreadable input raises ValueError with a readable message."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {name}")
    if path.lower().endswith(".epub"):
        try:
            raw = extract_epub(path)
        except Exception as e:  # ebooklib/zipfile raise a zoo of types for a bad EPUB
            raise ValueError(f"Could not read {name} as an EPUB: {e}") from e
        return preprocess(raw)
    try:
        with open(path, encoding="utf-8") as f:
            return preprocess(f.read())
    except UnicodeDecodeError as e:
        raise ValueError(f"{name} isn't UTF-8 text. Save it as UTF-8, or use an EPUB.") from e
