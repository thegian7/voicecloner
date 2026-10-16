"""Score generated audio against held-out real recordings of the speaker.

usage: eval_sim.py --root DIR SAMPLE_DIR [SAMPLE_DIR ...]

The targets are the val clips build_manifest.py set aside (never trained on), averaged per
source, so you can see whether a voice sounds like the speaker on streams and on calls.
Each SAMPLE_DIR is a folder of wavs (e.g. sample.py output for two checkpoints, or a plain
clone). Higher is closer. Treat it as a tie-breaker and always listen too.
"""
import argparse, glob, json, os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=lambda p: Path(p).expanduser())
ap.add_argument("dirs", nargs="+")
ap.add_argument("--gpu-gb", type=float, default=4)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_gb / total_gb))
spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(args.root / "models" / "ecapa"), run_opts={"device": dev})


@torch.no_grad()
def emb(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = torch.from_numpy(a.mean(1))
    if sr != 16000:
        a = AF.resample(a, sr, 16000)
    return torch.nn.functional.normalize(spk.encode_batch(a[None].to(dev)).squeeze(), dim=-1).cpu()


def unit_mean(vecs):
    return torch.nn.functional.normalize(torch.stack(vecs).mean(0), dim=-1)


val = [json.loads(l) for l in open(args.root / "data" / "val.jsonl")]
sources = sorted({v["source"] for v in val})
tgt = {s: unit_mean([emb(v["audio"]) for v in val if v["source"] == s]) for s in sources}
print("held-out clips:", {s: sum(v["source"] == s for v in val) for s in sources})
for d in args.dirs:
    wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
    if not wavs:
        print(f"{d}: no wavs")
        continue
    E = torch.stack([emb(p) for p in wavs])
    print(f"{os.path.basename(os.path.normpath(d)):20s}",
          "  ".join(f"{s}: {(E @ t).mean():.3f}" for s, t in tgt.items()), f"({len(wavs)} wavs)")
