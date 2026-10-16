# Training a speaker LoRA

A plain clone copies a voice from one 15-30 s clip. A LoRA fine-tuned on a few hours of the
same person talking naturally gets much closer, and the app uses it automatically once it is
attached to a saved voice (see "Speaker LoRA" in the main README).

These scripts take a pile of recordings, keep only the clips where the target speaker is
talking, and train a LoRA with VoxCPM's own trainer.

**The data never goes in git.** Training dirs hold real people's voices and transcripts.
Keep them outside the repo (the defaults below put them in `~/voicecloner-train/` on the
training box). `.gitignore` also blocks `*.flac`, `*.pcm` and any `data/`, `runs/`,
`models/` or `hf/` under `training/` in case a root lands in here.

If the voice isn't yours, get the person's OK before training on it.

## What you need

- 1-3 hours of the speaker talking naturally. Solo recordings (streams, voice memos,
  screencasts, the mic-only track of calls) are the best sources. Mixed recordings work
  but yield fewer usable minutes.
- One clean 15-30 s reference clip of only that speaker, lively and at their normal pace.
  It is used twice: to score every training clip, and later as the voice's reference in the
  app. Pick it carefully; VoxCPM copies its pace.
- A CUDA machine with:
  - a Python env with `voxcpm` installed (default `~/voicecloner/.venv`, set `VC_PYTHON`),
    used for training and samples;
  - a clone of https://github.com/OpenBMB/VoxCPM (default `~/VoxCPM`, set `VOXCPM_DIR`);
    the trainer lives in its `scripts/`;
  - a separate env for data prep (`pip install -r training/requirements-prep.txt` after
    installing torch);
  - `HF_HOME` pointing somewhere writable (default `~/voicecloner-train/hf`).

## Memory limits

Every script caps its CUDA allocator (`--gpu-gb`, `VC_GPU_GB`), and `train.sh` also runs the
job in a systemd scope with `MemoryMax=24G` and no swap when systemd is available, so a
runaway job is killed instead of squeezing the machine. That matters on unified-memory boxes
where GPU memory is host memory. `train_capped.py` casts the frozen base model to bf16, which
brings an r32 LoRA's peak to about 7 GB (stock training peaks near 28 GB).

## Steps

The examples train a voice called `alice` on a host called `gpubox`. `R` is its training dir.

```bash
R=voicecloner-train/alice         # relative to ~ on gpubox
```

**1. Ingest recordings** (on the machine that has the recordings and ffmpeg). The second
argument labels the source. Reports and held-out scores are split by it.

```bash
training/ingest.sh gpubox:$R streams ~/Movies/alice-stream-*.mp4
training/ingest.sh gpubox:$R calls ~/calls/             # finds every mic.pcm
scp alice-ref.wav gpubox:$R/ref.wav                     # the reference clip
```

**2. Find and score clips** (on gpubox). VAD splits each file into 2-15 s clips, Whisper
transcribes them, and ECAPA scores each against the reference. Already-scored files are
skipped, so rerun it after ingesting more.

```bash
P=~/voicecloner-train/.venv-prep/bin/python     # the data-prep env
HF_HOME=~/voicecloner-train/hf systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
  nice -n 10 $P training/prep_segments.py --root ~/$R --ref ~/$R/ref.wav
```

**3. Pick the training set.** `report` writes nothing; it prints minutes per file and
source at thresholds 0.45-0.60. Take the highest threshold that still leaves 1-3 h.

```bash
$P training/build_manifest.py --root ~/$R report
$P training/build_manifest.py --root ~/$R build 0.55
```

**4. Train.** It takes a while, so run it detached. Checkpoints land every 250 steps in
`runs/<run>/ckpt/step_NNNNNNN`.

```bash
nohup training/train.sh ~/$R lora1 1000 >/dev/null 2>&1 &
tail -f ~/$R/runs/lora1/train.out
```

If the app on that machine already downloaded VoxCPM2, point `VC_BASE_MODEL` at that
snapshot dir instead of downloading another copy.

**5. Listen and compare.** Render a few lines from a checkpoint with the reference (how the
app uses it), then score it against the held-out clips build_manifest set aside.

```bash
~/voicecloner/.venv/bin/python training/sample.py lines.txt ~/$R/samples/lora1 \
  ~/$R/runs/lora1/ckpt/step_0001000 --ref ~/$R/ref.wav --ref-text ref.txt
$P training/eval_sim.py --root ~/$R ~/$R/samples/lora1 ~/$R/samples/lora2
```

`lines.txt` holds one sentence per line, and `ref.txt` holds the reference clip's
transcript. `sample.py` makes 3 takes per line (cfg 1.6, 20 steps). Pick by ear and treat
the score as a tie-breaker.

**6. Use it.** Copy the checkpoint to the machine running the app and attach it to a saved
voice whose reference is the same clip:

```bash
scp -r gpubox:$R/runs/lora1/ckpt/step_0001000 /tmp/alice-lora
python -c 'from core.voices import attach_lora; attach_lora("Alice", "/tmp/alice-lora")'
```

## Continuing and retraining

- **More steps on the same data:** rerun with a higher total, for example
  `training/train.sh ~/$R lora1 2000`. The trainer resumes from `ckpt/latest` (weights,
  optimizer and schedule state). The cosine schedule is recomputed for the new total, so
  the learning rate jumps back up at the resume point, which works as a warm restart.
  Compare the new checkpoint with the old one before switching.
- **New recordings:** ingest them, rerun steps 2-3 (only new files are scored), and train
  a fresh run (`lora2`) rather than resuming, because the training set changed. Compare with
  `eval_sim.py`, then attach the winner.
- **Another voice:** use a new training dir and a new reference clip. Nothing is shared
  between voices except the downloaded models.

## Files

| File | Runs on | Does |
|---|---|---|
| `ingest.sh` | machine with the recordings | recordings → 16 kHz mono FLAC in `<root>/data/raw/<source>/` |
| `prep_segments.py` | training box | VAD → clips → Whisper text → speaker score; `data/segments.jsonl` |
| `build_manifest.py` | training box | `report` minutes by threshold; `build` cuts `data/clips/`, writes `train.jsonl` / `val.jsonl` |
| `train.sh` | training box | writes `runs/<run>/run.yaml` from `run.template.yaml`, launches inside the limits |
| `train_capped.py` | training box | VoxCPM's trainer + CUDA cap + bf16 frozen base |
| `sample.py` | training box | render test lines from a checkpoint |
| `eval_sim.py` | training box | score sample dirs against held-out real clips |
