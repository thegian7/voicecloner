"""
VoiceCloner — Unified app with Quick Clone, Audiobook, and My Voices tabs.
Run: python app.py
Then open http://localhost:7860
"""

import gc
import os
import re
import tempfile

import torch
import soundfile as sf
import gradio as gr

from core import (
    DEFAULT_STYLE,
    ModelCache,
    create_clone_prompt,
    generate_single,
    generate_full,
    generate_preview,
    prepare_text,
    save_wav,
    stitch_to_file,
    export_audio,
    preprocess,
    preprocess_file,
    save_voice,
    list_voices,
    load_voice,
    delete_voice,
)
from audiobook import session

cache = ModelCache()

TEMP_DIR = tempfile.mkdtemp(prefix="voicecloner_")

# ── Shared helpers ───────────────────────────────────────────────────


def _resolve_audio(ref_audio) -> str:
    """Convert gradio audio input (tuple or path) to a file path."""
    if isinstance(ref_audio, tuple):
        sr_in, audio_data = ref_audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR)
        sf.write(tmp.name, audio_data, sr_in)
        return tmp.name
    return ref_audio


def _export(wav_path: str, fmt: str) -> str:
    """Export a WAV file to the requested format, return output path."""
    if fmt == "wav":
        return wav_path
    ext = fmt if fmt != "m4a" else "m4a"
    out = wav_path.rsplit(".", 1)[0] + f".{ext}"
    return export_audio(wav_path, out, fmt)


# ── Voice library helpers ────────────────────────────────────────────


def _voice_choices() -> list[str]:
    return ["(New voice)"] + [v.name for v in list_voices()]


def _on_load_voice(voice_name: str):
    """When a saved voice is selected, populate the fields."""
    if voice_name == "(New voice)" or not voice_name:
        return gr.update(), gr.update(), gr.update(), gr.update()
    v = load_voice(voice_name)
    return (
        gr.update(value=v.ref_audio_path),
        gr.update(value=v.ref_text),
        gr.update(value=v.style),
        gr.update(value=v.use_design),
    )


# ── Quick Clone tab functions ────────────────────────────────────────


def generate_speech(text, ref_audio, ref_text, style, use_design, fmt):
    if not text.strip():
        raise gr.Error("Please enter text to speak.")
    if ref_audio is None:
        raise gr.Error("Please record or upload reference audio.")

    ref_audio_path = _resolve_audio(ref_audio)
    style_arg = style if (use_design and style.strip()) else ""

    voice_clone_prompt = create_clone_prompt(cache.clone, ref_audio_path, ref_text, style=style_arg)
    audio, sr = generate_full(text, cache.clone, voice_clone_prompt)

    out_path = os.path.join(TEMP_DIR, "quick_clone_output.wav")
    save_wav(out_path, audio, sr)
    return _export(out_path, fmt)


def preview_speech(text, ref_audio, ref_text, style, use_design):
    if not text.strip():
        raise gr.Error("Please enter text to speak.")
    if ref_audio is None:
        raise gr.Error("Please record or upload reference audio.")

    ref_audio_path = _resolve_audio(ref_audio)
    style_arg = style if (use_design and style.strip()) else ""

    voice_clone_prompt = create_clone_prompt(cache.clone, ref_audio_path, ref_text, style=style_arg)
    audio, sr = generate_preview(text, cache.clone, voice_clone_prompt)

    out_path = os.path.join(TEMP_DIR, "preview.wav")
    save_wav(out_path, audio, sr)
    return out_path


def load_text_file(file_path):
    """Load a .txt file into the text input."""
    if file_path is None:
        return gr.update()
    try:
        with open(file_path, encoding="utf-8") as f:
            return gr.update(value=f.read())
    except UnicodeDecodeError:
        raise gr.Error("That file isn't UTF-8 text. Save it as UTF-8 and try again.")
    except OSError as e:
        raise gr.Error(f"Could not read the file: {e.strerror or e}")


# ── Audiobook tab functions (from audiobook.py) ──────────────────────


def load_book_file(file_path):
    if file_path is None:
        raise gr.Error("Please upload a file.")
    try:
        chapters = preprocess_file(file_path)
    except ValueError as e:
        raise gr.Error(str(e))
    if not chapters:
        raise gr.Error("No text could be extracted from the file.")
    labels = [f"{i+1}. {ch.title}" for i, ch in enumerate(chapters)]
    return (
        chapters,
        gr.update(choices=labels, value=labels, visible=True),
        gr.update(visible=True),
    )


def load_pasted_text(book_text):
    if not book_text.strip():
        raise gr.Error("Please paste your book text first.")
    chapters = preprocess(book_text)
    if not chapters:
        raise gr.Error("No text could be extracted.")
    labels = [f"{i+1}. {ch.title}" for i, ch in enumerate(chapters)]
    return (
        chapters,
        gr.update(choices=labels, value=labels, visible=True),
        gr.update(visible=True),
    )


def chunk_selected_chapters(chapters, selected):
    if not selected:
        raise gr.Error("Select at least one chapter.")
    # labels look like "3. Chapter Title"; titles can contain dots and digits too
    selected_indices = {
        int(m.group(1)) - 1 for m in (re.match(r"^(\d+)\.", s) for s in selected) if m
    }
    selected_indices = {i for i in selected_indices if 0 <= i < len(chapters)}
    if not selected_indices:
        raise gr.Error("None of the selected chapters exist any more. Reload the text.")
    all_chunks = []
    for idx in sorted(selected_indices):
        ch = chapters[idx]
        ch_chunks = prepare_text(ch.text)
        for c in ch_chunks:
            all_chunks.append({"chapter": ch.title, "raw": c})

    state = [
        {
            "text": c["raw"].strip(),
            "audio_path": None,
            "approved": False,
            "is_para_end": c["raw"].endswith("\n\n"),
            "chapter": c["chapter"],
        }
        for c in all_chunks
    ]
    session.start()  # a new book replaces the last saved session
    session.save(state)
    current_ch = None
    lines = []
    for i, s in enumerate(state):
        if s["chapter"] != current_ch:
            current_ch = s["chapter"]
            lines.append(f"\n### {current_ch}\n")
        lines.append(f"**[{i+1}]** {s['text'][:80]}...")
    chunk_display = "\n---\n".join(lines)
    return state, chunk_display, gr.update(visible=True), gr.update(visible=True)


def setup_voice(ref_audio, ref_text, style, use_design):
    if ref_audio is None:
        raise gr.Error("Please upload reference audio.")
    ref_audio_path = _resolve_audio(ref_audio)
    style_arg = style if (use_design and style.strip()) else ""
    prompt = create_clone_prompt(cache.clone, ref_audio_path, ref_text, style=style_arg)
    return prompt, "Voice setup complete. Ready to generate."


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _generate_chunk(state, chunk, voice_prompt, name):
    """Generate one chunk into the session dir, then save the session for crash recovery.

    GPU memory is released even if generation fails.
    """
    try:
        audio, sr = generate_single(cache.clone, chunk["text"], voice_prompt)
        path = session.chunk_path(name)
        save_wav(path, audio, sr)
        del audio
    finally:
        _free_gpu()
    chunk["audio_path"] = path
    chunk["approved"] = False
    session.save(state)


def generate_all(state, voice_prompt, progress=gr.Progress()):
    if voice_prompt is None:
        raise gr.Error("Please set up your voice first (Step 2).")
    if not state:
        raise gr.Error("Please chunk your text first (Step 1).")

    results = []
    for i, chunk in enumerate(progress.tqdm(state, desc="Generating chunks")):
        if chunk["audio_path"] and chunk["approved"]:
            results.append(chunk)
            continue
        _generate_chunk(state, chunk, voice_prompt, f"chunk_{i:04d}.wav")
        results.append(chunk)

    return results, _build_review_md(results)


def regenerate_chunk(state, voice_prompt, chunk_idx):
    if voice_prompt is None:
        raise gr.Error("Set up voice first.")
    idx = int(chunk_idx)
    chunk = state[idx]
    _generate_chunk(state, chunk, voice_prompt, f"chunk_{idx:04d}_regen.wav")
    return state, _build_review_md(state)


def toggle_approve(state, chunk_idx):
    idx = int(chunk_idx)
    state[idx]["approved"] = not state[idx]["approved"]
    session.save(state)
    return state, _build_review_md(state)


def approve_all(state):
    for chunk in state:
        if chunk["audio_path"]:
            chunk["approved"] = True
    session.save(state)
    return state, _build_review_md(state)


def resume_session():
    saved = session.load()
    if saved is None or not saved[0]:
        raise gr.Error("No saved session to resume.")
    state = saved[0]
    return (state, gr.update(value=session.summary(state), visible=True),
            gr.update(visible=True), _build_review_md(state))


def _build_review_md(state):
    lines = []
    for i, c in enumerate(state):
        status = "Approved" if c["approved"] else ("Generated" if c["audio_path"] else "Pending")
        icon = {"Approved": "ok", "Generated": ">>", "Pending": ".."}[status]
        lines.append(f"[{icon}] **[{i+1}]** {status} -- {c['text'][:80]}...")
    return "\n\n".join(lines)


def export_audiobook(state, fmt):
    approved = [c for c in state if c["approved"] and c["audio_path"]]
    if not approved:
        raise gr.Error("No approved chunks to export. Generate and approve chunks first.")

    audio_paths = [c["audio_path"] for c in approved]
    paragraph_flags = [c["is_para_end"] for c in approved]
    wav_path = os.path.join(TEMP_DIR, "audiobook_final.wav")

    _, duration = stitch_to_file(audio_paths, wav_path, paragraph_flags=paragraph_flags)
    out = _export(wav_path, fmt)
    return out, f"Exported {len(approved)} chunks -- {duration:.1f}s total"


def audiobook_preview(ref_audio, ref_text, style, use_design):
    """Quick preview of the voice setup for audiobook tab."""
    if ref_audio is None:
        raise gr.Error("Please upload reference audio.")
    ref_audio_path = _resolve_audio(ref_audio)
    style_arg = style if (use_design and style.strip()) else ""
    prompt = create_clone_prompt(cache.clone, ref_audio_path, ref_text, style=style_arg)
    preview_text = "This is a preview of your voice clone."
    audio, sr = generate_single(cache.clone, preview_text, prompt)
    out_path = os.path.join(TEMP_DIR, "ab_preview.wav")
    save_wav(out_path, audio, sr)
    return out_path


# ── My Voices tab functions ──────────────────────────────────────────


def save_current_voice(name, ref_audio, ref_text, style, use_design):
    if not name.strip():
        raise gr.Error("Please enter a name for this voice.")
    if ref_audio is None:
        raise gr.Error("Please provide reference audio first.")
    ref_audio_path = _resolve_audio(ref_audio)
    save_voice(name, ref_audio_path, ref_text, style, use_design)
    return (
        f"Saved voice '{name}'.",
        gr.update(choices=_voice_choices()),
        gr.update(choices=_voice_choices()),
        _render_voices_list(),
    )


def delete_voice_action(name):
    if not name or name == "(New voice)":
        raise gr.Error("Select a voice to delete.")
    delete_voice(name)
    return (
        f"Deleted voice '{name}'.",
        gr.update(choices=_voice_choices()),
        gr.update(choices=_voice_choices()),
        _render_voices_list(),
    )


def _render_voices_list():
    voices = list_voices()
    if not voices:
        return "No saved voices yet. Save one from the Quick Clone or Audiobook tab."
    lines = []
    for v in voices:
        lines.append(
            f"**{v.name}** -- Style: {v.style[:50]}... | "
            f"Design: {'Yes' if v.use_design else 'No'} | {v.created_at[:10]}"
        )
    return "\n\n".join(lines)


# ── Build UI ─────────────────────────────────────────────────────────


with gr.Blocks(title="VoiceCloner") as app:
    gr.Markdown("# VoiceCloner\nClone your voice locally with VoxCPM2. No data leaves your machine.")

    with gr.Tab("Quick Clone"):
        with gr.Row():
            with gr.Column(scale=1):
                qc_voice_dropdown = gr.Dropdown(
                    label="Load saved voice",
                    choices=_voice_choices(),
                    value="(New voice)",
                )
                gr.Markdown("### 1. Your Voice")
                qc_ref_audio = gr.Audio(
                    label="Record or upload your voice (5-10 seconds)",
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                qc_ref_text = gr.Textbox(
                    label="What did you say? (optional, improves quality)",
                    placeholder="Type the transcript of your recording...",
                    lines=2,
                )
                gr.Markdown("### 2. Style (optional)")
                qc_use_design = gr.Checkbox(label="Apply style prompt", value=True)
                qc_style = gr.Textbox(
                    label="Describe the voice style",
                    value=DEFAULT_STYLE,
                    lines=2,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 3. Generate")
                qc_text = gr.Textbox(
                    label="Text to speak",
                    placeholder="Type what you want the clone to say...",
                    lines=4,
                )
                qc_txt_file = gr.File(
                    label="Or drop a .txt file",
                    file_types=[".txt"],
                )
                qc_format = gr.Dropdown(
                    label="Output format",
                    choices=["wav", "mp3", "flac", "m4a"],
                    value="wav",
                )
                with gr.Row():
                    qc_preview_btn = gr.Button("Preview", variant="secondary")
                    qc_gen_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                qc_preview_audio = gr.Audio(label="Preview", type="filepath")
                qc_output_audio = gr.Audio(label="Output", type="filepath")

        # Quick Clone wiring
        qc_voice_dropdown.change(
            fn=_on_load_voice,
            inputs=[qc_voice_dropdown],
            outputs=[qc_ref_audio, qc_ref_text, qc_style, qc_use_design],
        )
        qc_txt_file.change(
            fn=load_text_file,
            inputs=[qc_txt_file],
            outputs=[qc_text],
        )
        qc_preview_btn.click(
            fn=preview_speech,
            inputs=[qc_text, qc_ref_audio, qc_ref_text, qc_style, qc_use_design],
            outputs=qc_preview_audio,
        )
        qc_gen_btn.click(
            fn=generate_speech,
            inputs=[qc_text, qc_ref_audio, qc_ref_text, qc_style, qc_use_design, qc_format],
            outputs=qc_output_audio,
        )

    with gr.Tab("Audiobook"):
        ab_voice_prompt_state = gr.State(None)
        ab_chunks_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                ab_voice_dropdown = gr.Dropdown(
                    label="Load saved voice",
                    choices=_voice_choices(),
                    value="(New voice)",
                )
                gr.Markdown("### Step 1: Voice Setup")
                ab_ref_audio = gr.Audio(
                    label="Reference audio (5-10s of your voice)",
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                ab_ref_text = gr.Textbox(label="Reference transcript (optional)", lines=2)
                ab_use_design = gr.Checkbox(label="Apply style prompt", value=True)
                ab_style = gr.Textbox(label="Voice style", value=DEFAULT_STYLE, lines=2)
                with gr.Row():
                    ab_setup_btn = gr.Button("Set Up Voice", variant="primary")
                    ab_preview_btn = gr.Button("Preview Voice", variant="secondary")
                ab_setup_status = gr.Textbox(label="Status", interactive=False)
                ab_preview_audio = gr.Audio(label="Voice Preview", type="filepath")

            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Load Book Text")
                with gr.Tab("Upload File"):
                    ab_file_input = gr.File(
                        label="Upload EPUB or TXT",
                        file_types=[".epub", ".txt"],
                    )
                    ab_upload_btn = gr.Button("Load File", variant="primary")
                with gr.Tab("Paste Text"):
                    ab_book_text = gr.Textbox(
                        label="Full text",
                        placeholder="Paste your book text here...",
                        lines=10,
                    )
                    ab_paste_btn = gr.Button("Load Text", variant="primary")

        ab_chapters_state = gr.State([])
        ab_chapter_selector = gr.CheckboxGroup(
            label="Select chapters to include",
            choices=[],
            visible=False,
        )
        with gr.Row():
            ab_chunk_btn = gr.Button("Chunk Selected Chapters", variant="primary", visible=False)
            ab_resume_btn = gr.Button("Resume Last Session", variant="secondary")
        ab_chunk_display = gr.Markdown("", visible=False)

        with gr.Group(visible=False) as ab_gen_group:
            gr.Markdown("### Step 3: Generate & Review")
            with gr.Row():
                ab_gen_all_btn = gr.Button("Generate All", variant="primary")
                ab_approve_all_btn = gr.Button("Approve All")

            ab_review_display = gr.Markdown("")

            with gr.Row():
                ab_regen_idx = gr.Number(label="Chunk # to regenerate", precision=0, minimum=1)
                ab_regen_btn = gr.Button("Regenerate Chunk")
                ab_approve_idx = gr.Number(label="Chunk # to toggle approve", precision=0, minimum=1)
                ab_approve_btn = gr.Button("Toggle Approve")

            gr.Markdown("### Step 4: Export")
            ab_export_format = gr.Dropdown(
                label="Export format",
                choices=["wav", "mp3", "flac", "m4a"],
                value="wav",
            )
            ab_export_btn = gr.Button("Export Audiobook", variant="primary")
            ab_export_status = gr.Textbox(label="Export status", interactive=False)
            ab_export_audio = gr.Audio(label="Final Audiobook", type="filepath")

        # Audiobook wiring
        ab_voice_dropdown.change(
            fn=_on_load_voice,
            inputs=[ab_voice_dropdown],
            outputs=[ab_ref_audio, ab_ref_text, ab_style, ab_use_design],
        )
        ab_setup_btn.click(
            fn=setup_voice,
            inputs=[ab_ref_audio, ab_ref_text, ab_style, ab_use_design],
            outputs=[ab_voice_prompt_state, ab_setup_status],
        )
        ab_preview_btn.click(
            fn=audiobook_preview,
            inputs=[ab_ref_audio, ab_ref_text, ab_style, ab_use_design],
            outputs=ab_preview_audio,
        )
        ab_upload_btn.click(
            fn=load_book_file,
            inputs=[ab_file_input],
            outputs=[ab_chapters_state, ab_chapter_selector, ab_chunk_btn],
        )
        ab_paste_btn.click(
            fn=load_pasted_text,
            inputs=[ab_book_text],
            outputs=[ab_chapters_state, ab_chapter_selector, ab_chunk_btn],
        )
        ab_chunk_btn.click(
            fn=chunk_selected_chapters,
            inputs=[ab_chapters_state, ab_chapter_selector],
            outputs=[ab_chunks_state, ab_chunk_display, ab_chunk_display, ab_gen_group],
        )
        ab_resume_btn.click(
            fn=resume_session,
            inputs=[],
            outputs=[ab_chunks_state, ab_chunk_display, ab_gen_group, ab_review_display],
        )
        ab_gen_all_btn.click(
            fn=generate_all,
            inputs=[ab_chunks_state, ab_voice_prompt_state],
            outputs=[ab_chunks_state, ab_review_display],
        )
        ab_regen_btn.click(
            fn=lambda state, vp, idx: regenerate_chunk(state, vp, int(idx) - 1),
            inputs=[ab_chunks_state, ab_voice_prompt_state, ab_regen_idx],
            outputs=[ab_chunks_state, ab_review_display],
        )
        ab_approve_btn.click(
            fn=lambda state, idx: toggle_approve(state, int(idx) - 1),
            inputs=[ab_chunks_state, ab_approve_idx],
            outputs=[ab_chunks_state, ab_review_display],
        )
        ab_approve_all_btn.click(
            fn=approve_all,
            inputs=[ab_chunks_state],
            outputs=[ab_chunks_state, ab_review_display],
        )
        ab_export_btn.click(
            fn=export_audiobook,
            inputs=[ab_chunks_state, ab_export_format],
            outputs=[ab_export_audio, ab_export_status],
        )

    with gr.Tab("My Voices"):
        mv_voices_display = gr.Markdown(_render_voices_list())
        gr.Markdown("### Save Current Voice")
        mv_name = gr.Textbox(label="Voice name", placeholder="e.g. My Narrator Voice")
        mv_ref_audio = gr.Audio(
            label="Reference audio",
            type="filepath",
            sources=["microphone", "upload"],
        )
        mv_ref_text = gr.Textbox(label="Reference transcript (optional)", lines=2)
        mv_use_design = gr.Checkbox(label="Apply style prompt", value=True)
        mv_style = gr.Textbox(label="Voice style", value=DEFAULT_STYLE, lines=2)
        mv_save_btn = gr.Button("Save Voice", variant="primary")
        mv_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Delete Voice")
        mv_delete_dropdown = gr.Dropdown(
            label="Select voice to delete",
            choices=_voice_choices(),
        )
        mv_delete_btn = gr.Button("Delete Voice", variant="stop")

        # My Voices wiring
        mv_save_btn.click(
            fn=save_current_voice,
            inputs=[mv_name, mv_ref_audio, mv_ref_text, mv_style, mv_use_design],
            outputs=[mv_status, qc_voice_dropdown, ab_voice_dropdown, mv_voices_display],
        )
        mv_delete_btn.click(
            fn=delete_voice_action,
            inputs=[mv_delete_dropdown],
            outputs=[mv_status, qc_voice_dropdown, ab_voice_dropdown, mv_voices_display],
        )

if __name__ == "__main__":
    from core import get_device
    print(f"Device: {get_device()}")
    print("Models will load on first generation.")
    app.launch(theme=gr.themes.Soft())
