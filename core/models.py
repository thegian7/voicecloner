import torch
from qwen_tts import Qwen3TTSModel

from .device import get_device

DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class ModelCache:
    """Lazy-loads and keeps both models alive (for server/Gradio mode)."""

    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        self._design = None
        self._clone = None

    @property
    def design(self):
        if self._design is None:
            print(f"Loading VoiceDesign model on {self.device}...")
            self._design = Qwen3TTSModel.from_pretrained(
                DESIGN_MODEL_ID, device_map=self.device, dtype=torch.float16
            )
        return self._design

    @property
    def clone(self):
        if self._clone is None:
            print(f"Loading Clone model on {self.device}...")
            self._clone = Qwen3TTSModel.from_pretrained(
                CLONE_MODEL_ID, device_map=self.device, dtype=torch.float32
            )
        return self._clone


def load_design_model(device: str | None = None):
    """Standalone loader for CLI one-shot mode."""
    dev = device or get_device()
    return Qwen3TTSModel.from_pretrained(
        DESIGN_MODEL_ID, device_map=dev, dtype=torch.float16
    )


def load_clone_model(device: str | None = None):
    """Standalone loader for CLI one-shot mode."""
    dev = device or get_device()
    return Qwen3TTSModel.from_pretrained(
        CLONE_MODEL_ID, device_map=dev, dtype=torch.float32
    )


def free_model(model, device: str | None = None):
    """Delete model and free GPU memory."""
    del model
    dev = device or get_device()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev.startswith("cuda"):
        torch.cuda.empty_cache()
