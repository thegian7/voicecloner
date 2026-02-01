"""Chapter loading, chunking, and text editing for audiobook production."""

from __future__ import annotations

import gradio as gr

from core import preprocess, preprocess_file, prepare_text, Chapter
from .state import BookState, ChunkState


def load_file(file_path: str | None) -> tuple[list[Chapter], list[str]]:
    """Load an EPUB or TXT file and return (chapters, label_list)."""
    if file_path is None:
        raise gr.Error("Please upload a file.")
    chapters = preprocess_file(file_path)
    if not chapters:
        raise gr.Error("No text could be extracted from the file.")
    labels = [f"{i + 1}. {ch.title}" for i, ch in enumerate(chapters)]
    return chapters, labels


def load_pasted_text(book_text: str) -> tuple[list[Chapter], list[str]]:
    """Preprocess pasted text and return (chapters, label_list)."""
    if not book_text.strip():
        raise gr.Error("Please paste your book text first.")
    chapters = preprocess(book_text)
    if not chapters:
        raise gr.Error("No text could be extracted.")
    labels = [f"{i + 1}. {ch.title}" for i, ch in enumerate(chapters)]
    return chapters, labels


def chunk_chapters(
    chapters: list[Chapter],
    selected_labels: list[str],
    book: BookState,
) -> BookState:
    """Chunk selected chapters into BookState, applying pronunciation dict."""
    if not selected_labels:
        raise gr.Error("Select at least one chapter.")

    selected_indices = {int(s.split(".")[0]) - 1 for s in selected_labels}
    book.chunks = []

    for idx in sorted(selected_indices):
        ch = chapters[idx]
        text = book.apply_pronunciation(ch.text)
        ch_chunks = prepare_text(text)
        for raw in ch_chunks:
            book.chunks.append(
                ChunkState(
                    text=raw.strip(),
                    chapter=ch.title,
                    is_para_end=raw.endswith("\n\n"),
                )
            )
    return book


def edit_chunk_text(book: BookState, chunk_idx: int, new_text: str) -> BookState:
    """Update the text for a single chunk (clears its audio so it must be regenerated)."""
    idx = int(chunk_idx)
    if idx < 0 or idx >= len(book.chunks):
        raise gr.Error(f"Invalid chunk index: {idx + 1}")
    chunk = book.chunks[idx]
    new_text = new_text.strip()
    if new_text and new_text != chunk.text:
        chunk.text = new_text
        chunk.audio_path = None
        chunk.approved = False
    return book
