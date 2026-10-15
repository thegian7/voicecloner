"""Gradio UI for audiobook production.

Guided wizard: Voice Setup -> Load Text -> Review Chunks -> Generate & Review -> Export
"""

from __future__ import annotations

import gradio as gr

from .state import BookState, DEFAULT_STYLE
from . import voice as voice_mod
from . import chapters as ch_mod
from . import generate as gen_mod
from . import export as export_mod


# ---------------------------------------------------------------------------
# UI builder helpers
# ---------------------------------------------------------------------------

def _build_chapter_progress(book: BookState) -> str:
    """Markdown showing per-chapter progress bars."""
    if not book.chunks:
        return ""
    summary = book.progress_summary()
    lines: list[str] = []
    for ch_name, counts in summary.items():
        total = counts["pending"] + counts["generated"] + counts["approved"]
        done = counts["approved"]
        gen = counts["generated"]
        pct = int((done + gen) / total * 100) if total else 0
        status = "done" if done == total else "partial" if (done + gen) > 0 else "pending"
        icon = {"done": "✅", "partial": "🔄", "pending": "⏳"}[status]
        lines.append(
            f"{icon} **{ch_name}** — "
            f"{done}/{total} approved, {gen} generated, {counts['pending']} pending "
            f"({pct}%)"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Handler wrappers (bridge between Gradio events and business logic)
# ---------------------------------------------------------------------------

def _on_setup_voice(ref_audio, ref_text, style_text, use_design):
    prompt, status = voice_mod.setup_voice(ref_audio, ref_text, style_text, use_design)
    return prompt, status


def _on_preview_voice(voice_prompt, preview_text):
    path = voice_mod.preview_voice(voice_prompt, preview_text)
    return path


def _on_load_file(file_path):
    chapters, labels = ch_mod.load_file(file_path)
    return (
        chapters,
        gr.update(choices=labels, value=labels, visible=True),
        gr.update(visible=True),
    )


def _on_load_paste(book_text):
    chapters, labels = ch_mod.load_pasted_text(book_text)
    return (
        chapters,
        gr.update(choices=labels, value=labels, visible=True),
        gr.update(visible=True),
    )


def _on_chunk(chapters_list, selected, book: BookState, pronunciation_text):
    book.update_pronunciation(pronunciation_text)
    book = ch_mod.chunk_chapters(chapters_list, selected, book)
    progress_md = _build_chapter_progress(book)
    return book, progress_md, gr.update(visible=True)


def _on_generate_all(book: BookState, voice_prompt, progress=gr.Progress()):
    book = gen_mod.generate_all(book, voice_prompt, progress)
    return book, _build_chapter_progress(book), _render_review(book)


def _on_regenerate(book: BookState, voice_prompt, chunk_num):
    idx = int(chunk_num) - 1
    book = gen_mod.regenerate_chunk(book, voice_prompt, idx)
    return book, _build_chapter_progress(book), _render_review(book)


def _on_edit_chunk(book: BookState, chunk_num, new_text):
    idx = int(chunk_num) - 1
    book = ch_mod.edit_chunk_text(book, idx, new_text)
    return book, _build_chapter_progress(book), _render_review(book)


def _on_toggle_approve(book: BookState, chunk_num):
    idx = int(chunk_num) - 1
    book = gen_mod.toggle_approve(book, idx)
    return book, _build_chapter_progress(book), _render_review(book)


def _on_approve_all(book: BookState):
    book = gen_mod.approve_all(book)
    return book, _build_chapter_progress(book), _render_review(book)


def _on_export(book: BookState, fmt, per_chapter):
    result, status = export_mod.export_audiobook(book, fmt=fmt, per_chapter=per_chapter)
    if isinstance(result, list):
        # per-chapter export: return first file for the audio player, list all in status
        first = result[0] if result else None
        status += "\n\nFiles:\n" + "\n".join(result)
        return first, status
    return result, status


def _render_review(book: BookState) -> str:
    """Build the full review markdown with per-chunk details grouped by chapter."""
    if not book.chunks:
        return "*No chunks loaded yet.*"

    lines: list[str] = []
    current_ch = None

    for i, c in enumerate(book.chunks):
        if c.chapter != current_ch:
            current_ch = c.chapter
            lines.append(f"\n---\n### {current_ch}\n")

        if c.approved:
            icon = "✅"
            status = "Approved"
        elif c.audio_path:
            icon = "🔊"
            status = "Generated"
        else:
            icon = "⏳"
            status = "Pending"

        audio_link = ""
        if c.audio_path:
            audio_link = f"  \n*Audio: `{c.audio_path}`*"

        lines.append(
            f"{icon} **[{i + 1}]** {status} — {c.text[:120]}{'...' if len(c.text) > 120 else ''}"
            f"{audio_link}"
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main UI layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    with gr.Blocks(title="Audiobook Producer") as app:
        gr.Markdown("# Audiobook Producer\nVoice Setup → Load Text → Chunk → Generate → Export")

        # ----- shared state -----
        voice_prompt_state = gr.State(None)
        chapters_state = gr.State([])
        book_state = gr.State(BookState())

        # ===== Tab 1: Voice Setup =====
        with gr.Tab("1. Voice Setup"):
            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(
                        label="Reference audio (5-10s of your voice)",
                        type="filepath",
                        sources=["microphone", "upload"],
                    )
                    ref_text = gr.Textbox(
                        label="Reference transcript (optional — improves quality)",
                        lines=2,
                    )
                    use_design = gr.Checkbox(label="Use VoiceDesign for style", value=True)
                    style = gr.Textbox(label="Voice style description", value=DEFAULT_STYLE, lines=2)
                    setup_btn = gr.Button("Set Up Voice", variant="primary")
                    setup_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    gr.Markdown("### Preview Your Voice")
                    preview_text = gr.Textbox(
                        label="Preview text",
                        value="Hello! This is a preview of your cloned voice reading an audiobook.",
                        lines=2,
                    )
                    preview_btn = gr.Button("Generate Preview")
                    preview_audio = gr.Audio(label="Preview", type="filepath")

            setup_btn.click(
                fn=_on_setup_voice,
                inputs=[ref_audio, ref_text, style, use_design],
                outputs=[voice_prompt_state, setup_status],
            )
            preview_btn.click(
                fn=_on_preview_voice,
                inputs=[voice_prompt_state, preview_text],
                outputs=[preview_audio],
            )

        # ===== Tab 2: Load Text =====
        with gr.Tab("2. Load Text"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Upload File"):
                        file_input = gr.File(
                            label="Upload EPUB or TXT",
                            file_types=[".epub", ".txt"],
                        )
                        upload_btn = gr.Button("Load File", variant="primary")

                    with gr.Tab("Paste Text"):
                        book_text_input = gr.Textbox(
                            label="Full text",
                            placeholder="Paste your book text here...",
                            lines=12,
                        )
                        paste_btn = gr.Button("Load Text", variant="primary")

                with gr.Column():
                    gr.Markdown("### Pronunciation Dictionary")
                    gr.Markdown(
                        "Add custom pronunciations, one per line: `find -> replace`\n\n"
                        "These are applied **before** chunking so the TTS sees the corrected text."
                    )
                    pronunciation_box = gr.Textbox(
                        label="Pronunciation rules",
                        value=BookState().pronunciation_as_text(),
                        lines=8,
                    )

            chapter_selector = gr.CheckboxGroup(
                label="Select chapters to include",
                choices=[],
                visible=False,
            )
            chunk_btn = gr.Button("Chunk Selected Chapters", variant="primary", visible=False)

            upload_btn.click(
                fn=_on_load_file,
                inputs=[file_input],
                outputs=[chapters_state, chapter_selector, chunk_btn],
            )
            paste_btn.click(
                fn=_on_load_paste,
                inputs=[book_text_input],
                outputs=[chapters_state, chapter_selector, chunk_btn],
            )

        # ===== Tab 3: Generate & Review =====
        with gr.Tab("3. Generate & Review"):
            progress_display = gr.Markdown("*Load and chunk text first.*")

            with gr.Row():
                gen_all_btn = gr.Button("Generate All Chunks", variant="primary")
                approve_all_btn = gr.Button("Approve All")

            review_display = gr.Markdown("*No chunks yet.*")

            gr.Markdown("---")
            gr.Markdown("### Per-Chunk Actions")
            with gr.Row():
                with gr.Column():
                    action_chunk_num = gr.Number(
                        label="Chunk #", precision=0, minimum=1, value=1
                    )
                with gr.Column():
                    regen_btn = gr.Button("Regenerate This Chunk")
                    approve_btn = gr.Button("Toggle Approve")

            gr.Markdown("### Edit Chunk Text")
            gr.Markdown("Change the text for a chunk, then regenerate it.")
            with gr.Row():
                edit_chunk_num = gr.Number(label="Chunk #", precision=0, minimum=1, value=1)
                edit_text = gr.Textbox(label="New text", lines=3)
            edit_btn = gr.Button("Save Edit")

            # Wire chunk button from Tab 2 to update Tab 3
            chunk_btn.click(
                fn=_on_chunk,
                inputs=[chapters_state, chapter_selector, book_state, pronunciation_box],
                outputs=[book_state, progress_display, gen_all_btn],
            )

            gen_all_btn.click(
                fn=_on_generate_all,
                inputs=[book_state, voice_prompt_state],
                outputs=[book_state, progress_display, review_display],
            )
            regen_btn.click(
                fn=_on_regenerate,
                inputs=[book_state, voice_prompt_state, action_chunk_num],
                outputs=[book_state, progress_display, review_display],
            )
            approve_btn.click(
                fn=_on_toggle_approve,
                inputs=[book_state, action_chunk_num],
                outputs=[book_state, progress_display, review_display],
            )
            approve_all_btn.click(
                fn=_on_approve_all,
                inputs=[book_state],
                outputs=[book_state, progress_display, review_display],
            )
            edit_btn.click(
                fn=_on_edit_chunk,
                inputs=[book_state, edit_chunk_num, edit_text],
                outputs=[book_state, progress_display, review_display],
            )

        # ===== Tab 4: Export =====
        with gr.Tab("4. Export"):
            gr.Markdown("### Export Settings")
            with gr.Row():
                export_fmt = gr.Dropdown(
                    label="Format",
                    choices=[
                        ("WAV", "wav"), ("MP3", "mp3"), ("FLAC", "flac"), ("M4A", "m4a"),
                        ("ACX / Audible (MP3 per chapter, mastered + checked)", "acx"),
                    ],
                    value="wav",
                )
                export_per_chapter = gr.Checkbox(
                    label="Export per chapter (separate files)",
                    value=False,
                )

            export_btn = gr.Button("Export Audiobook", variant="primary")
            export_status = gr.Textbox(label="Export status", interactive=False)
            export_audio = gr.Audio(label="Final Audiobook", type="filepath")

            export_btn.click(
                fn=_on_export,
                inputs=[book_state, export_fmt, export_per_chapter],
                outputs=[export_audio, export_status],
            )

    return app
