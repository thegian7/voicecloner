"""Audio generation, regeneration, and approval logic."""

from __future__ import annotations

import gc
import os

import torch
import gradio as gr

from core import generate_single, save_wav
from .state import BookState, TEMP_DIR
from .voice import cache


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def generate_all(book: BookState, voice_prompt, progress=gr.Progress()) -> BookState:
    """Generate audio for every chunk that isn't already approved."""
    if voice_prompt is None:
        raise gr.Error("Please set up your voice first.")
    if not book.chunks:
        raise gr.Error("Please load and chunk your text first.")

    for i, chunk in enumerate(progress.tqdm(book.chunks, desc="Generating chunks")):
        if chunk.audio_path and chunk.approved:
            continue

        audio, sr = generate_single(cache.clone, chunk.text, voice_prompt)
        path = os.path.join(TEMP_DIR, f"chunk_{i:04d}.wav")
        save_wav(path, audio, sr)
        chunk.audio_path = path
        chunk.approved = False

        del audio
        _free_gpu_memory()

    return book


def generate_chapter(
    book: BookState, voice_prompt, chapter_name: str, progress=gr.Progress()
) -> BookState:
    """Generate audio for all unapproved chunks in a single chapter."""
    if voice_prompt is None:
        raise gr.Error("Please set up your voice first.")

    pairs = book.chunks_for_chapter(chapter_name)
    if not pairs:
        raise gr.Error(f"No chunks found for chapter: {chapter_name}")

    for gi, chunk in progress.tqdm(pairs, desc=f"Generating {chapter_name}"):
        if chunk.audio_path and chunk.approved:
            continue
        audio, sr = generate_single(cache.clone, chunk.text, voice_prompt)
        path = os.path.join(TEMP_DIR, f"chunk_{gi:04d}.wav")
        save_wav(path, audio, sr)
        chunk.audio_path = path
        chunk.approved = False
        del audio
        _free_gpu_memory()

    return book


def regenerate_chunk(book: BookState, voice_prompt, chunk_idx: int) -> BookState:
    """Regenerate a single chunk."""
    if voice_prompt is None:
        raise gr.Error("Set up voice first.")
    idx = int(chunk_idx)
    if idx < 0 or idx >= len(book.chunks):
        raise gr.Error(f"Invalid chunk index: {idx + 1}")

    chunk = book.chunks[idx]
    audio, sr = generate_single(cache.clone, chunk.text, voice_prompt)
    path = os.path.join(TEMP_DIR, f"chunk_{idx:04d}_regen.wav")
    save_wav(path, audio, sr)
    del audio
    _free_gpu_memory()

    chunk.audio_path = path
    chunk.approved = False
    return book


def toggle_approve(book: BookState, chunk_idx: int) -> BookState:
    """Toggle approval on a single chunk."""
    idx = int(chunk_idx)
    if idx < 0 or idx >= len(book.chunks):
        raise gr.Error(f"Invalid chunk index: {idx + 1}")
    book.chunks[idx].approved = not book.chunks[idx].approved
    return book


def approve_all(book: BookState) -> BookState:
    """Approve every chunk that has audio."""
    for chunk in book.chunks:
        if chunk.audio_path:
            chunk.approved = True
    return book


def approve_chapter(book: BookState, chapter_name: str) -> BookState:
    """Approve all generated chunks in a chapter."""
    for _, chunk in book.chunks_for_chapter(chapter_name):
        if chunk.audio_path:
            chunk.approved = True
    return book
