import re

CHUNK_TARGET_MAX = 200


def prepare_text(raw_text: str, chunk_max: int = CHUNK_TARGET_MAX) -> list[str]:
    """Split raw text into generation-friendly chunks with pacing punctuation."""
    text = raw_text.strip()
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for pi, para in enumerate(paragraphs):
        para = " ".join(para.split())
        if not para:
            continue

        # Breathing pauses: insert ... after commas in long clauses
        para = re.sub(r",(?=\s[^.!?]{60,})", ",...", para)

        sentences = re.split(r"(?<=[.!?])\s+", para)

        current = ""
        for sent in sentences:
            if not sent:
                continue
            if current and len(current) + len(sent) + 1 > chunk_max:
                chunks.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent

        if current:
            chunks.append(current.strip())

        # Mark paragraph boundary (except after last paragraph)
        if pi < len(paragraphs) - 1 and chunks:
            chunks[-1] = chunks[-1] + "\n\n"

    return chunks
