"""Pass 1: VAD -> 2-15 s clips -> Whisper text -> ECAPA similarity to the target speaker's reference.

Reads every FLAC under <root>/data/raw/<source>/ (see ingest.sh) and writes
<root>/data/segments.jsonl, one row per candidate clip with its transcript and its speaker
similarity to --ref. Nothing is cut or kept yet; pass 2 (build_manifest.py) picks a
threshold from the score distribution and writes the training clips.
Resumable: files already present in segments.jsonl are skipped, so adding recordings
later only processes the new ones.

usage: prep_segments.py --root ~/voicecloner-train/<voice> --ref <voice>-ref.wav
"""
import argparse, json, time
from pathlib import Path

import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from speechbrain.inference.speaker import EncoderClassifier
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--root", required=True, type=lambda p: Path(p).expanduser(),
                help="training dir for this voice (holds data/raw/)")
ap.add_argument("--ref", required=True, type=lambda p: Path(p).expanduser(),
                help="15-30 s clip of ONLY the target speaker; every clip is scored against it")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--gpu-gb", type=float, default=8,
                help="hard cap on the CUDA allocator (default 8 GB)")
ap.add_argument("--language", default="en", help="Whisper transcription language")
args = ap.parse_args()

ROOT = args.root
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "segments.jsonl"
REF = args.ref
SR = 16000
MIN_S, MAX_S, JOIN_GAP = 2.0, 15.0, 0.45

dev = args.device
if dev.startswith("cuda"):
    # On a unified-memory box every GPU byte is a byte the host and its other
    # services lose. Cap the allocator so this job errors out on its own before it can
    # squeeze the machine.
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_gb / total_gb))
vad = load_silero_vad()
spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(ROOT / "models" / "ecapa"), run_opts={"device": dev})
# Whisper called directly: the transformers pipeline drags in torchcodec, which has no
# FFmpeg libs on every box. Clips are <= 15 s, so one 30 s window each is enough.
WHISPER = "openai/whisper-large-v3-turbo"
asr_proc = WhisperProcessor.from_pretrained(WHISPER)
ASR_DTYPE = torch.float16 if dev.startswith("cuda") else torch.float32
asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=ASR_DTYPE).to(dev).eval()


@torch.no_grad()
def transcribe(chunks):
    out = []
    for i in range(0, len(chunks), 8):
        feats = asr_proc([c.numpy() for c in chunks[i:i + 8]], sampling_rate=SR, return_tensors="pt")
        ids = asr_model.generate(feats.input_features.to(dev, ASR_DTYPE), language=args.language, task="transcribe")
        out += asr_proc.batch_decode(ids, skip_special_tokens=True)
    return out


def load(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    a = torch.from_numpy(a)
    if sr != SR:
        import torchaudio.functional as AF
        a = AF.resample(a, sr, SR)
    return a


@torch.no_grad()
def embed(chunks):
    lens = torch.tensor([len(c) for c in chunks], dtype=torch.float32)
    pad = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
    e = spk.encode_batch(pad.to(dev), lens / lens.max())
    return torch.nn.functional.normalize(e.squeeze(1), dim=-1).cpu()


ref = load(REF)
# reference embedding: mean of 4 s windows, so it is comparable to clip-length audio
win = [ref[i:i + 4 * SR] for i in range(0, len(ref) - 4 * SR, 2 * SR)]
ref_e = torch.nn.functional.normalize(embed(win).mean(0), dim=-1)

done = set()
if OUT.exists():
    done = {json.loads(l)["file"] for l in OUT.open()}

files = sorted(p for p in RAW.rglob("*.flac"))
for path in files:
    rel = str(path.relative_to(RAW))
    if rel in done:
        continue
    t_start = time.time()
    audio = load(path)
    ts = get_speech_timestamps(audio, vad, sampling_rate=SR, min_silence_duration_ms=250,
                               speech_pad_ms=80, return_seconds=True)
    # merge VAD regions into clips that end on a pause
    clips, cur = [], None
    for r in ts:
        if cur and r["start"] - cur[1] <= JOIN_GAP and r["end"] - cur[0] <= MAX_S:
            cur[1] = r["end"]
        else:
            if cur:
                clips.append(cur)
            cur = [r["start"], r["end"]]
    if cur:
        clips.append(cur)
    clips = [c for c in clips if MIN_S <= c[1] - c[0] <= MAX_S]

    rows = []
    for i in range(0, len(clips), 8):
        batch = clips[i:i + 8]
        chunks = [audio[int(a * SR):int(b * SR)] for a, b in batch]
        sims = (embed(chunks) @ ref_e).tolist()
        texts = transcribe(chunks)
        for (a, b), s, t, c in zip(batch, sims, texts, chunks):
            rows.append({"file": rel, "source": rel.split("/")[0], "t0": round(a, 3), "t1": round(b, 3),
                         "sim": round(s, 4), "rms": round(float(c.pow(2).mean().sqrt()), 5),
                         "text": t.strip()})
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if not rows:
            f.write(json.dumps({"file": rel, "source": rel.split("/")[0], "empty": True}) + "\n")
    del audio
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    speech = sum(r["t1"] - r["t0"] for r in rows)
    print(f"{rel}: {len(rows)} clips, {speech/60:.1f} min speech, {time.time()-t_start:.0f}s", flush=True)
print("DONE", flush=True)
