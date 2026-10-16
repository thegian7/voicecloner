"""Pass 2: report usable minutes per source, then cut kept clips + write train/val manifests.

usage: build_manifest.py --root DIR report            # minutes per file/source at several thresholds, writes nothing
       build_manifest.py --root DIR build THRESHOLD   # cuts data/clips/*.wav, writes data/train.jsonl, data/val.jsonl

THRESHOLD is the speaker similarity to the reference a clip needs to be kept. Read the
report first: pick the highest threshold that still leaves 1-3 hours. Around 0.55 is a good
start; below ~0.5 other people start leaking in.
"""
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=lambda p: Path(p).expanduser())
ap.add_argument("cmd", choices=("report", "build"))
ap.add_argument("threshold", nargs="?", type=float)
args = ap.parse_args()
if args.cmd == "build" and args.threshold is None:
    ap.error("build needs a THRESHOLD (run report first)")

ROOT = args.root
RAW = ROOT / "data" / "raw"
SEG = ROOT / "data" / "segments.jsonl"
CLIPS = ROOT / "data" / "clips"

rows = [json.loads(l) for l in SEG.open()]
rows = [r for r in rows if "sim" in r]

HALLUCINATIONS = re.compile(r"^(thank you\.?|thanks for watching.*|you\.?|bye\.?|\W*)$", re.I)


def clean(r):
    """Text/audio sanity independent of who is speaking."""
    t = r["text"]
    words = len(t.split())
    dur = r["t1"] - r["t0"]
    if words < 3 or HALLUCINATIONS.match(t.strip()):
        return False
    wps = words / dur
    if not 1.2 <= wps <= 4.8:  # too slow = mostly silence/music, too fast = hallucinated text
        return False
    if re.search(r"\b(\w+)( \1\b){3,}", t, re.I):  # "the the the the"
        return False
    return r["rms"] > 0.003


def file_label(f):
    return f.split("/", 1)[1].rsplit(".", 1)[0]


def report():
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    ths = (0.45, 0.5, 0.55, 0.6)
    print(f"{'file':48s} {'speech':>7s} {'clean':>7s} " + " ".join(f"≥{t:.2f}" for t in ths))
    tot = defaultdict(float)
    for f in sorted(by_file):
        rs = by_file[f]
        sp = sum(r["t1"] - r["t0"] for r in rs) / 60
        cl = [r for r in rs if clean(r)]
        cm = sum(r["t1"] - r["t0"] for r in cl) / 60
        per = [sum(r["t1"] - r["t0"] for r in cl if r["sim"] >= t) / 60 for t in ths]
        src = f.split("/")[0]
        tot[(src, "speech")] += sp
        tot[(src, "clean")] += cm
        for t, m in zip(ths, per):
            tot[(src, t)] += m
        print(f"{file_label(f):48s} {sp:7.1f} {cm:7.1f} " + " ".join(f"{m:5.1f}" for m in per))
    print()
    sources = sorted({s for s, _ in tot})
    for src in sources:
        print(f"{src:48s} {tot[(src,'speech')]:7.1f} {tot[(src,'clean')]:7.1f} "
              + " ".join(f"{tot[(src,t)]:5.1f}" for t in ths))
    print(f"{'TOTAL (minutes)':48s} {sum(tot[(s,'speech')] for s in sources):7.1f} "
          f"{sum(tot[(s,'clean')] for s in sources):7.1f} "
          + " ".join(f"{sum(tot[(s,t)] for s in sources):5.1f}" for t in ths))


def build(th):
    CLIPS.mkdir(parents=True, exist_ok=True)
    keep = [r for r in rows if clean(r) and r["sim"] >= th]
    if len(keep) < 2:
        raise SystemExit(f"only {len(keep)} clip(s) pass {th}; lower the threshold or add recordings")
    n_val = min(max(20, len(keep) // 20), len(keep) // 5 or 1)  # 5% held out, at least 20 when there's room
    val_ids = set(random.Random(17).sample(range(len(keep)), n_val))
    cache = {}
    out = {"val": [], "train": []}
    for i, r in enumerate(keep):  # rows are grouped by file, so each source is read once
        if r["file"] not in cache:
            cache.clear()
            cache[r["file"]] = sf.read(str(RAW / r["file"]), dtype="float32")
        audio, sr = cache[r["file"]]
        name = f"{file_label(r['file'])}_{int(r['t0']*1000):08d}.wav"
        sf.write(str(CLIPS / name), audio[int(r["t0"] * sr):int(r["t1"] * sr)], sr, subtype="PCM_16")
        out["val" if i in val_ids else "train"].append(
            {"audio": str(CLIPS / name), "text": r["text"], "source": r["source"], "sim": r["sim"]})
    for k, v in out.items():
        with (ROOT / "data" / f"{k}.jsonl").open("w") as f:
            for x in v:
                f.write(json.dumps(x) + "\n")
    mins = {k: sum(sf.info(x["audio"]).duration for x in v) / 60 for k, v in out.items()}
    print(f"threshold {th}: train {len(out['train'])} clips / {mins['train']:.1f} min, "
          f"val {len(out['val'])} clips / {mins['val']:.1f} min")


if args.cmd == "report":
    report()
else:
    build(args.threshold)
