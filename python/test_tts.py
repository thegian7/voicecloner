"""Quick test of Qwen3-TTS on Apple Silicon (MPS)."""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# --- VoiceDesign: generate speech with a built-in speaker ---
print("\n=== VoiceDesign Test (1.7B) ===")
print("Loading model (will download on first run)...")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype=torch.float16,
)

wavs, sr = model.generate_custom_voice(
    text="Hello! This is a test of the Qwen 3 text to speech system running locally on Apple Silicon.",
    language="English",
    speaker="Vivian",
)

sf.write("output_design.wav", wavs[0], sr)
print(f"Saved output_design.wav ({len(wavs[0])/sr:.1f}s, {sr}Hz)")

del model
torch.mps.empty_cache()

# --- VoiceClone: clone from a reference audio ---
# Uncomment below if you have a reference audio file
# print("\n=== VoiceClone Test (0.6B) ===")
# print("Loading clone model...")
# clone_model = Qwen3TTSModel.from_pretrained(
#     "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceClone",
#     device_map="mps",
#     dtype=torch.float32,  # MUST be float32 for clone on MPS
# )
# wavs, sr = clone_model.generate_voice_clone(
#     text="This is cloned speech.",
#     language="English",
#     reference_audio="reference.wav",
# )
# sf.write("output_clone.wav", wavs[0], sr)
# print(f"Saved output_clone.wav ({len(wavs[0])/sr:.1f}s, {sr}Hz)")

print("\nDone!")
