"""Export / stitching logic for audiobook production."""

from __future__ import annotations

import os

import gradio as gr

from core import stitch_to_file, export_audio
from .state import BookState, TEMP_DIR


def export_audiobook(
    book: BookState,
    fmt: str = "wav",
    per_chapter: bool = False,
) -> tuple[str | list[str], str]:
    """Stitch approved chunks and export.

    Returns (file_path_or_list, status_message).
    """
    approved = [c for c in book.chunks if c.approved and c.audio_path]
    if not approved:
        raise gr.Error("No approved chunks to export. Generate and approve chunks first.")

    if per_chapter:
        return _export_per_chapter(book, fmt)
    return _export_single(book, approved, fmt)


def _export_single(book: BookState, approved, fmt: str) -> tuple[str, str]:
    audio_paths = [c.audio_path for c in approved if c.audio_path]
    paragraph_flags = [c.is_para_end for c in approved]
    wav_path = os.path.join(TEMP_DIR, "audiobook_final.wav")

    sr, duration = stitch_to_file(audio_paths, wav_path, paragraph_flags=paragraph_flags)

    if fmt == "wav":
        return wav_path, f"Exported {len(approved)} chunks — {duration:.1f}s total"

    out_path = os.path.join(TEMP_DIR, f"audiobook_final.{fmt}")
    export_audio(wav_path, out_path, fmt=fmt)
    return out_path, f"Exported {len(approved)} chunks as {fmt.upper()} — {duration:.1f}s total"


def _export_per_chapter(book: BookState, fmt: str) -> tuple[list[str], str]:
    paths: list[str] = []
    total_duration = 0.0

    for ch_name in book.chapter_names:
        ch_approved = [
            c for c in book.chunks
            if c.chapter == ch_name and c.approved and c.audio_path
        ]
        if not ch_approved:
            continue

        audio_paths = [c.audio_path for c in ch_approved if c.audio_path]
        paragraph_flags = [c.is_para_end for c in ch_approved]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in ch_name).strip()
        wav_path = os.path.join(TEMP_DIR, f"{safe_name}.wav")

        sr, dur = stitch_to_file(audio_paths, wav_path, paragraph_flags=paragraph_flags)
        total_duration += dur

        if fmt != "wav":
            out_path = os.path.join(TEMP_DIR, f"{safe_name}.{fmt}")
            export_audio(wav_path, out_path, fmt=fmt)
            paths.append(out_path)
        else:
            paths.append(wav_path)

    if not paths:
        raise gr.Error("No approved chapters to export.")

    msg = f"Exported {len(paths)} chapters as {fmt.upper()} — {total_duration:.1f}s total"
    return paths, msg
