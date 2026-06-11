import torch
from voxcpm import VoxCPM

from .device import get_device

TTS_MODEL_ID = "openbmb/VoxCPM2"

# Legacy aliases — VoxCPM2 collapses the old Qwen3 clone + design models into one
CLONE_MODEL_ID = TTS_MODEL_ID
DESIGN_MODEL_ID = TTS_MODEL_ID


def _safe_device(device: str) -> str | None:
    # VoxCPM2 issue #301: mps + bfloat16 emits noise on macOS. Let VoxCPM auto-pick
    # (it falls back to CPU on darwin) until the upstream bug is fixed.
    if device == "mps":
        return None
    return device


class ModelCache:
    """Lazy-loads and keeps the VoxCPM2 model alive (for server/Gradio mode).

    VoxCPM2 is a single model that handles both voice cloning (reference audio)
    and voice design (natural-language style prompt). `cache.clone` and
    `cache.design` both return the same underlying model — kept as aliases for
    backward compatibility with callers from the Qwen3-TTS era.
    """

    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        self._tts = None

    @property
    def tts(self):
        if self._tts is None:
            print(f"Loading VoxCPM2 (device={self.device})...")
            self._tts = VoxCPM.from_pretrained(
                TTS_MODEL_ID,
                load_denoiser=False,
                device=_safe_device(self.device),
            )
        return self._tts

    @property
    def clone(self):
        return self.tts

    @property
    def design(self):
        return self.tts


def load_tts_model(device: str | None = None):
    """Standalone loader for CLI one-shot mode."""
    dev = device or get_device()
    return VoxCPM.from_pretrained(
        TTS_MODEL_ID,
        load_denoiser=False,
        device=_safe_device(dev),
    )


# Legacy aliases for callers that still import the Qwen3-era names
load_clone_model = load_tts_model
load_design_model = load_tts_model


def free_model(model, device: str | None = None):
    """Delete model and free GPU memory."""
    del model
    dev = device or get_device()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev.startswith("cuda"):
        torch.cuda.empty_cache()
