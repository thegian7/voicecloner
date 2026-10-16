import gc
import json
import os

import torch
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model

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


def use_lora(model, lora_dir: str | None) -> None:
    """Point a loaded VoxCPM at a voice's LoRA, or switch LoRA off when lora_dir is empty.

    VoxCPM only has LoRA layers if it was built with a LoRAConfig. Models start
    without them (the extra matmuls slow every voice down on CPU), so the first
    LoRA voice rebuilds the inner model in place, with the same weights and device.
    After that, switching voices only swaps the small LoRA weights or toggles them.
    """
    tm = model.tts_model
    if not lora_dir:
        if tm.lora_config is not None:
            model.set_lora_enabled(False)
        return

    with open(os.path.join(lora_dir, "lora_config.json"), encoding="utf-8") as f:
        cfg = json.load(f).get("lora_config", {})
    if tm.lora_config is None or tm.lora_config.model_dump() != LoRAConfig(**cfg).model_dump():
        print(f"Rebuilding VoxCPM2 with LoRA layers (r={cfg.get('r')})...")
        device = str(tm.device)
        base = snapshot_download(TTS_MODEL_ID)
        model.tts_model = None
        del tm
        gc.collect()
        model.tts_model = VoxCPM2Model.from_local(base, optimize=False, device=device,
                                                  lora_config=LoRAConfig(**cfg))
        model._vc_lora = None
    if getattr(model, "_vc_lora", None) != lora_dir:
        loaded, skipped = model.load_lora(lora_dir)
        if not loaded:
            raise RuntimeError(f"No LoRA weights from {lora_dir} matched the model ({len(skipped)} skipped).")
        model._vc_lora = lora_dir
    model.set_lora_enabled(True)


def free_model(model, device: str | None = None):
    """Delete model and free GPU memory."""
    del model
    dev = device or get_device()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev.startswith("cuda"):
        torch.cuda.empty_cache()
