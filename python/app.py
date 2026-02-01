"""
VoiceCloner — Clone your voice with Qwen3-TTS
Run: python app.py
Then open http://localhost:7860
"""

import torch
import soundfile as sf
import gradio as gr
import tempfile
from qwen_tts import Qwen3TTSModel

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Global model cache
design_model = None
clone_model = None


def load_models():
    global design_model, clone_model

    if design_model is None:
        print(f"Loading VoiceDesign model on {DEVICE}...")
        design_model = Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL_ID, device_map=DEVICE, dtype=torch.float16
        )

    if clone_model is None:
        print(f"Loading Clone model on {DEVICE}...")
        clone_model = Qwen3TTSModel.from_pretrained(
            CLONE_MODEL_ID, device_map=DEVICE, dtype=torch.float32
        )


def generate_speech(
    text: str,
    ref_audio,
    ref_text: str,
    style: str,
    use_design: bool,
):
    if not text.strip():
        raise gr.Error("Please enter text to speak.")

    if ref_audio is None:
        raise gr.Error("Please record or upload reference audio.")

    load_models()

    # Handle gradio audio input (sr, numpy array) or file path
    if isinstance(ref_audio, tuple):
        sr_in, audio_data = ref_audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_data, sr_in)
        ref_audio_path = tmp.name
    else:
        ref_audio_path = ref_audio

    # Step 1: VoiceDesign (optional)
    if use_design and style.strip():
        design_model.generate_voice_design(
            text="Hello everybody, hope you are having an awesome day!",
            language="English",
            instruct=style,
        )

    # Step 2: Build clone prompt from reference
    voice_clone_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text if ref_text.strip() else "",
        x_vector_only_mode=not ref_text.strip(),
    )

    # Step 3: Generate
    wavs, sr = clone_model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out_path, wavs[0], sr)
    return out_path


DEFAULT_STYLE = (
    "A young excited American male with a warm energetic tone, "
    "speaking with enthusiasm and confidence like a podcast host"
)

with gr.Blocks(title="VoiceCloner") as app:
    gr.Markdown("# VoiceCloner\nClone your voice locally with Qwen3-TTS. No data leaves your machine.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Your Voice")
            ref_audio = gr.Audio(
                label="Record or upload your voice (5-10 seconds)",
                type="filepath",
                sources=["microphone", "upload"],
            )
            ref_text = gr.Textbox(
                label="What did you say? (optional, improves quality)",
                placeholder="Type the transcript of your recording...",
                lines=2,
            )

            gr.Markdown("### 2. Style (optional)")
            use_design = gr.Checkbox(label="Use VoiceDesign for style", value=True)
            style = gr.Textbox(
                label="Describe the voice style",
                value=DEFAULT_STYLE,
                lines=2,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 3. Generate")
            text = gr.Textbox(
                label="Text to speak",
                placeholder="Type what you want the clone to say...",
                lines=4,
            )
            btn = gr.Button("Generate Speech", variant="primary", size="lg")
            output_audio = gr.Audio(label="Output", type="filepath")

    btn.click(
        fn=generate_speech,
        inputs=[text, ref_audio, ref_text, style, use_design],
        outputs=output_audio,
    )

if __name__ == "__main__":
    import os

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(f"Device: {DEVICE}")
    print("Models will load on first generation.")
    app.launch(server_port=port, theme=gr.themes.Soft())
