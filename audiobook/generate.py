"""Audio generation, regeneration, and approval logic.

Chunk audio is written to the session dir and the chunk list is saved after every
change, so an interrupted run can be resumed (see session.py).
"""

from __future__ import annotations

import gc

import torch
import gradio as gr

from core import generate_single, save_wav
from . import session
from .state import BookState, ChunkState
from .voice import cache


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _generate_into(book: BookState, chunk: ChunkState, voice_prompt, name: str) -> None:
    """Generate one chunk's audio into the session dir and save the session.

    GPU memory is released even when generation fails, so one bad chunk doesn't leave
    the next attempt short of memory.
    """
    try:
        audio, sr = generate_single(cache.clone, chunk.text, voice_prompt)
        path = session.chunk_path(name)
        save_wav(path, audio, sr)
        del audio
    finally:
        _free_gpu_memory()
    chunk.audio_path = path
    chunk.approved = False
    book.save()


def generate_all(book: BookState, voice_prompt, progress=gr.Progress()) -> BookState:
    """Generate audio for every chunk that isn't already approved."""
    if voice_prompt is None:
        raise ValueError("Please set up your voice first.")
    if not book.chunks:
        raise ValueError("Please load and chunk your text first.")

    for i, chunk in enumerate(progress.tqdm(book.chunks, desc="Generating chunks")):
        if chunk.audio_path and chunk.approved:
            continue
        _generate_into(book, chunk, voice_prompt, f"chunk_{i:04d}.wav")

    return book


def generate_chapter(
    book: BookState, voice_prompt, chapter_name: str, progress=gr.Progress()
) -> BookState:
    """Generate audio for all unapproved chunks in a single chapter."""
    if voice_prompt is None:
        raise ValueError("Please set up your voice first.")

    pairs = book.chunks_for_chapter(chapter_name)
    if not pairs:
        raise ValueError(f"No chunks found for chapter: {chapter_name}")

    for gi, chunk in progress.tqdm(pairs, desc=f"Generating {chapter_name}"):
        if chunk.audio_path and chunk.approved:
            continue
        _generate_into(book, chunk, voice_prompt, f"chunk_{gi:04d}.wav")

    return book


def regenerate_chunk(book: BookState, voice_prompt, chunk_idx: int) -> BookState:
    """Regenerate a single chunk."""
    if voice_prompt is None:
        raise ValueError("Set up voice first.")
    idx = int(chunk_idx)
    if idx < 0 or idx >= len(book.chunks):
        raise ValueError(f"Invalid chunk index: {idx + 1}")

    _generate_into(book, book.chunks[idx], voice_prompt, f"chunk_{idx:04d}_regen.wav")
    return book


def toggle_approve(book: BookState, chunk_idx: int) -> BookState:
    """Toggle approval on a single chunk."""
    idx = int(chunk_idx)
    if idx < 0 or idx >= len(book.chunks):
        raise ValueError(f"Invalid chunk index: {idx + 1}")
    book.chunks[idx].approved = not book.chunks[idx].approved
    book.save()
    return book


def approve_all(book: BookState) -> BookState:
    """Approve every chunk that has audio."""
    for chunk in book.chunks:
        if chunk.audio_path:
            chunk.approved = True
    book.save()
    return book


def approve_chapter(book: BookState, chapter_name: str) -> BookState:
    """Approve all generated chunks in a chapter."""
    for _, chunk in book.chunks_for_chapter(chapter_name):
        if chunk.audio_path:
            chunk.approved = True
    book.save()
    return book
