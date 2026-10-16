"""Render test lines from a LoRA checkpoint, a few takes each, for listening and eval_sim.py.

usage: sample.py LINES.txt OUTDIR CKPT_DIR [--ref REF.wav --ref-text REF.txt]

LINES.txt holds one sentence per line (blank lines and lines starting with # are skipped).
With --ref, the LoRA is used the way the app uses it: the voice's reference clip as
prompt + reference ("prompt" mode, usually the best-sounding setup). Without
it the voice has to come from the weights alone ("bare" mode), which is a harder test of
what the LoRA learned.

Writes OUTDIR/<line>_t<take>.wav and OUTDIR/manifest.json. Existing wavs are kept, so an
interrupted run picks up where it stopped.
"""
import argparse, json, os, sys

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("lines")
ap.add_argument("outdir")
ap.add_argument("ckpt", help="a step_NNNNNNN dir holding lora_weights.safetensors + lora_config.json")
ap.add_argument("--ref", help="reference clip of the speaker (prompt mode)")
ap.add_argument("--ref-text", help="file holding the reference clip's transcript")
ap.add_argument("--takes", type=int, default=3)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=1.6)
ap.add_argument("--gpu-gb", type=float, default=10)
args = ap.parse_args()
if args.ref and not args.ref_text:
    ap.error("--ref needs --ref-text (the clip's transcript)")

if torch.cuda.is_available():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_gb / total_gb))

sys.path.insert(0, os.path.join(os.path.expanduser(os.environ.get("VOXCPM_DIR", "~/VoxCPM")), "src"))
from voxcpm.core import VoxCPM  # noqa: E402
from voxcpm.model.voxcpm2 import LoRAConfig  # noqa: E402

lines = [l.strip() for l in open(args.lines, encoding="utf-8") if l.strip() and not l.startswith("#")]
ref_text = open(args.ref_text, encoding="utf-8").read().strip() if args.ref else None

info = json.load(open(os.path.join(args.ckpt, "lora_config.json")))
model = VoxCPM.from_pretrained(hf_model_id=info["base_model"], load_denoiser=False, optimize=False,
                               lora_config=LoRAConfig(**info["lora_config"]), lora_weights_path=args.ckpt)
sr = model.tts_model.sample_rate
os.makedirs(args.outdir, exist_ok=True)
kw = dict(prompt_wav_path=args.ref, prompt_text=ref_text, reference_wav_path=args.ref) if args.ref else {}
manifest = []
for i, text in enumerate(lines):
    for k in range(args.takes):
        path = os.path.join(args.outdir, f"l{i:02d}_t{k}.wav")
        if not os.path.exists(path):
            torch.manual_seed(1000 + 17 * k + i)
            a = model.generate(text=text, cfg_value=args.cfg, inference_timesteps=args.steps, **kw)
            sf.write(path, np.asarray(a, dtype=np.float32), sr)
        manifest.append({"line": i, "take": k, "text": text, "path": path})
        print(f"l{i:02d} t{k}", flush=True)
json.dump(manifest, open(os.path.join(args.outdir, "manifest.json"), "w"), indent=0)
