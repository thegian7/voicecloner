"""
VoiceCloner — Clone your voice locally with VoxCPM2.
Quick Clone + My Voices (no audiobook features).
Run: python app.py
"""

import os
import tempfile

import soundfile as sf
import gradio as gr

from core import (
    DEFAULT_STYLE,
    ModelCache,
    create_clone_prompt,
    generate_full,
    generate_preview,
    save_wav,
    export_audio,
    save_voice,
    list_voices,
    load_voice,
    delete_voice,
)

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
    with open(file_path, encoding="utf-8") as f:
        return gr.update(value=f.read())


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
        _render_voices_list(),
    )


def delete_voice_action(name):
    if not name or name == "(New voice)":
        raise gr.Error("Select a voice to delete.")
    delete_voice(name)
    return (
        f"Deleted voice '{name}'.",
        gr.update(choices=_voice_choices()),
        _render_voices_list(),
    )


def _render_voices_list():
    voices = list_voices()
    if not voices:
        return "No saved voices yet. Save one from the Quick Clone tab."
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
            outputs=[mv_status, qc_voice_dropdown, mv_voices_display],
        )
        mv_delete_btn.click(
            fn=delete_voice_action,
            inputs=[mv_delete_dropdown],
            outputs=[mv_status, qc_voice_dropdown, mv_voices_display],
        )

if __name__ == "__main__":
    from core import get_device

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(f"Device: {get_device()}")
    print("Models will load on first generation.")
    app.launch(server_port=port, theme=gr.themes.Soft())
