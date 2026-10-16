"""Run VoxCPM's own trainer with a hard CUDA memory cap. Normally started by train.sh.

On a unified-memory box the GPU memory is the host's memory, so training gets VC_GPU_GB
(default 14; train.sh passes 12) and errors out by itself rather than squeezing the host.
The r32 LoRA on VoxCPM2 peaks at ~7.3 GB with this wrapper; stock it peaks near 28 GB.

In training mode VoxCPM keeps the whole frozen base model in fp32 (~9 GB). Only the
LoRA adapters train, so the frozen base weights are cast to bf16 (autocast computes in
bf16 anyway); LoRA params and the AudioVAE stay fp32.

usage: train_capped.py --config_path run.yaml
env:   VOXCPM_DIR  a checkout of github.com/OpenBMB/VoxCPM (default ~/VoxCPM); its
                   scripts/train_voxcpm_finetune.py is the trainer
       VC_GPU_GB   CUDA allocator cap in GB
       VC_BF16_BASE=0 keeps the frozen base in fp32
"""
import atexit
import os
import runpy
import sys

import torch

cap_gb = float(os.environ.get("VC_GPU_GB", "14"))
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb / total_gb))
print(f"[train_capped] CUDA allocator capped at {cap_gb:.0f} GB of {total_gb:.0f} GB", file=sys.stderr)
atexit.register(lambda: print(f"[train_capped] peak allocated "
                              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", file=sys.stderr))

VOXCPM_DIR = os.path.expanduser(os.environ.get("VOXCPM_DIR", "~/VoxCPM"))
sys.path.insert(0, os.path.join(VOXCPM_DIR, "src"))
from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: E402

_from_local = VoxCPM2Model.from_local.__func__


def _from_local_bf16_base(cls, *args, **kwargs):
    model = _from_local(cls, *args, **kwargs)
    if kwargs.get("training") and os.environ.get("VC_BF16_BASE", "1") == "1":
        n = 0
        for name, p in model.named_parameters():
            if not p.requires_grad and not name.startswith("audio_vae") and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
                n += p.numel()
        print(f"[train_capped] frozen base -> bf16 ({n / 1e9:.2f}B params)", file=sys.stderr)
    return model


VoxCPM2Model.from_local = classmethod(_from_local_bf16_base)

script = os.path.join(VOXCPM_DIR, "scripts", "train_voxcpm_finetune.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
