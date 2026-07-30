"""Cast-aware audiobook generation CLI.

Reads a tagged script + a cast manifest, generates one WAV per script line
using the right voice, then stitches per-chapter WAVs with crossfades.
Resumable — skips lines whose WAV already exists.

Output layout:
    <out_dir>/<chapter_slug>/<line_slug>.wav        per-line audio
    <out_dir>/<chapter_slug>.wav                    stitched chapter

CLI:
    python -m audiobook.run \\
        --script /path/to/script-dir-or-file \\
        --cast   /path/to/cast.toml \\
        --out    /path/to/output

    [--chapter "Chapter 2"]      only this chapter (substring match on name)
    [--export wav|mp3|flac|m4a]  per-chapter export format (default wav)
    [--no-stitch]                generate per-line WAVs but skip chapter stitch
    [--force]                    regenerate even if WAV already exists
    [--workers N]                generate N lines concurrently (default 1)
    [--restitch]                 re-stitch existing line WAVs, generate nothing
    [--no-normalize]             skip loudness matching at stitch time
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
import time

# Add parent dir so `from core import ...` works when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    DEFAULT_PEAK_DBFS,
    DEFAULT_TARGET_DBFS,
    ModelCache,
    create_clone_prompt,
    export_audio,
    generate_full,
    lin_to_dbfs,
    plan_gains,
    save_wav,
    speech_rms,
    stitch_to_file,
)

from .cast import Cast, load_cast, validate_cast_against_voices, validate_cast_against_script
from .script import ScriptLine, parse_script_path, speakers_in, chapters_in


def _slug(name: str) -> str:
    """Filesystem-safe slug — keep alphanumeric, replace others with -."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return s or "unnamed"


def _voice_prompt_for(cast: Cast, speaker: str, _cache: dict[str, dict]) -> dict:
    """Resolve speaker → voice profile → voice prompt dict (memoised per-run)."""
    if speaker in _cache:
        return _cache[speaker]
    profile = cast.voice_profile(speaker)
    prompt = create_clone_prompt(
        None,  # model arg is unused
        ref_audio=profile.ref_audio_path,
        ref_text=profile.ref_text or "",
        style="",  # never apply style with a reference audio (gets read aloud)
    )
    _cache[speaker] = prompt
    return prompt


def plan_paths(lines: list[ScriptLine], out_dir: str) -> dict[str, list[str]]:
    """Where each line's WAV lives, without generating anything.

    Same layout generate_lines() writes to. Lets --restitch rebuild a chapter
    from existing line WAVs without paying the model load.
    """
    by_chapter: dict[str, list[str]] = {}
    for line in lines:
        chapter_dir = os.path.join(out_dir, _slug(line.chapter))
        by_chapter.setdefault(line.chapter, []).append(
            os.path.join(chapter_dir, f"{line.slug}.wav")
        )
    return by_chapter


def _generate_shard(payload: tuple) -> dict:
    """Worker entry point: generate an assigned subset of lines.

    Runs in its own process with its own model copy (~4.6GB), so this trades
    memory for throughput. Safe to parallelise because every line writes to a
    distinct path — no coordination needed — and stitch order is recovered
    separately from the script, not from completion order.

    Must stay module-level and take only picklable args: mp uses 'spawn' here
    (required for torch on macOS), so the payload is re-pickled per worker.
    """
    tag, cast_path, out_dir, force, lines = payload
    cast = load_cast(cast_path)
    model_cache = ModelCache()
    _ = model_cache.tts  # eager load
    prompt_cache: dict[str, dict] = {}

    stats = {"done": 0, "skipped": 0, "failed": 0, "audio_s": 0.0, "gen_s": 0.0}
    total = len(lines)
    for idx, line in enumerate(lines, start=1):
        chapter_dir = os.path.join(out_dir, _slug(line.chapter))
        os.makedirs(chapter_dir, exist_ok=True)
        wav_path = os.path.join(chapter_dir, f"{line.slug}.wav")

        if not force and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            stats["skipped"] += 1
            continue

        try:
            prompt = _voice_prompt_for(cast, line.speaker, prompt_cache)
        except KeyError as e:
            print(f"{tag}[{idx:4d}/{total}] !!! skipping (no voice): {e}", flush=True)
            stats["failed"] += 1
            continue

        t0 = time.time()
        try:
            audio, sr = generate_full(line.text, model_cache.tts, prompt)
            save_wav(wav_path, audio, sr)
            dur = len(audio) / sr
            elapsed = time.time() - t0
            stats["done"] += 1
            stats["audio_s"] += dur
            stats["gen_s"] += elapsed
            print(
                f"{tag}[{idx:4d}/{total}] {line.speaker:10s} "
                f"{dur:5.1f}s audio / {elapsed:5.1f}s gen "
                f"({elapsed/max(dur,0.01):.2f}x RT) — {line.text[:44]}...",
                flush=True,
            )
        except Exception as e:
            print(
                f"{tag}[{idx:4d}/{total}] FAILED ({type(e).__name__}: {e}) "
                f"— {line.text[:60]}",
                flush=True,
            )
            stats["failed"] += 1
    return stats


def generate_lines_parallel(
    lines: list[ScriptLine],
    cast_path: str,
    out_dir: str,
    workers: int,
    force: bool = False,
) -> dict[str, list[str]]:
    """Generate per-line WAVs across `workers` processes.

    Lines are dealt round-robin rather than in contiguous blocks so every worker
    sees a similar mix of long narration and short dialogue — contiguous blocks
    would leave one worker holding a run of 30-second narrator lines while
    another burns through one-word replies.
    """
    todo = [
        ln for ln in lines
        if force or not (
            os.path.exists(os.path.join(out_dir, _slug(ln.chapter), f"{ln.slug}.wav"))
            and os.path.getsize(os.path.join(out_dir, _slug(ln.chapter), f"{ln.slug}.wav")) > 0
        )
    ]
    already = len(lines) - len(todo)
    if already:
        print(f"Resuming: {already} line(s) already rendered, {len(todo)} to go.")
    if not todo:
        print("All lines already rendered — nothing to generate.")
        return plan_paths(lines, out_dir)

    shards = [todo[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]
    print(
        f"Generating {len(todo)} line(s) across {len(shards)} worker(s): "
        f"{[len(s) for s in shards]}"
    )

    payloads = [
        (f"w{i} ", cast_path, out_dir, force, s) for i, s in enumerate(shards)
    ]
    t0 = time.time()
    if len(payloads) == 1:
        results = [_generate_shard(payloads[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(_generate_shard, payloads)
    wall = time.time() - t0

    done = sum(r["done"] for r in results)
    failed = sum(r["failed"] for r in results)
    audio_s = sum(r["audio_s"] for r in results)
    print(
        f"\nGenerated {done} line(s) ({audio_s/60:.1f} min audio) in {wall/60:.1f} min "
        f"= {audio_s/max(wall,0.01):.2f}x realtime"
        + (f" — {failed} FAILED" if failed else "")
    )
    if failed:
        print("Re-run the same command to retry only the failures (existing WAVs are skipped).")
    return plan_paths(lines, out_dir)


def generate_lines(
    lines: list[ScriptLine],
    cast: Cast,
    out_dir: str,
    force: bool = False,
) -> dict[str, list[str]]:
    """Generate per-line WAVs serially. Returns {chapter_name: [wav_path_in_order]}."""
    model_cache = ModelCache()
    _ = model_cache.tts  # eager load
    prompt_cache: dict[str, dict] = {}

    by_chapter: dict[str, list[str]] = {}

    total = len(lines)
    for idx, line in enumerate(lines, start=1):
        chapter_dir = os.path.join(out_dir, _slug(line.chapter))
        os.makedirs(chapter_dir, exist_ok=True)
        wav_path = os.path.join(chapter_dir, f"{line.slug}.wav")
        by_chapter.setdefault(line.chapter, []).append(wav_path)

        if not force and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"[{idx:4d}/{total}] skip (exists) {line.speaker}: {line.text[:60]}...")
            continue

        try:
            prompt = _voice_prompt_for(cast, line.speaker, prompt_cache)
        except KeyError as e:
            print(f"[{idx:4d}/{total}] !!! skipping (no voice): {e}")
            continue

        t0 = time.time()
        try:
            audio, sr = generate_full(line.text, model_cache.tts, prompt)
            save_wav(wav_path, audio, sr)
            dur = len(audio) / sr
            elapsed = time.time() - t0
            print(
                f"[{idx:4d}/{total}] {line.speaker:10s} "
                f"{dur:5.1f}s audio / {elapsed:5.1f}s gen "
                f"({elapsed/max(dur,0.01):.2f}x RT) — {line.text[:50]}..."
            )
        except Exception as e:
            print(f"[{idx:4d}/{total}] FAILED ({type(e).__name__}: {e}) — {line.text[:60]}")

    return by_chapter


def stitch_chapters(
    by_chapter: dict[str, list[str]],
    out_dir: str,
    export_format: str = "wav",
    normalize: bool = True,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    peak_dbfs: float = DEFAULT_PEAK_DBFS,
) -> None:
    """Stitch per-chapter WAVs into one file per chapter."""
    for chapter, wav_paths in by_chapter.items():
        existing = [p for p in wav_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        if not existing:
            print(f"  [stitch] no WAVs for '{chapter}', skipping")
            continue

        gains = None
        if normalize:
            gains = plan_gains(existing, target_dbfs=target_dbfs, peak_dbfs=peak_dbfs)
            adjust = [abs(lin_to_dbfs(g)) for g in gains]
            print(
                f"  [norm] {chapter}: {len(gains)} lines to {target_dbfs:.1f} dBFS RMS "
                f"/ {peak_dbfs:.1f} dBFS peak "
                f"(gain {lin_to_dbfs(min(gains)):+.1f} to {lin_to_dbfs(max(gains)):+.1f} dB, "
                f"median move {sorted(adjust)[len(adjust)//2]:.1f} dB)"
            )

        stitched = os.path.join(out_dir, f"{_slug(chapter)}.wav")
        # paragraph_flags=None gives all-standard pauses; works for line-level output
        sr, dur = stitch_to_file(existing, stitched, paragraph_flags=None, gains=gains)
        print(f"  [stitch] {chapter}: {len(existing)} lines -> {stitched} ({dur:.1f}s @ {sr}Hz)")

        if export_format != "wav":
            out_path = stitched.rsplit(".", 1)[0] + f".{export_format}"
            export_audio(stitched, out_path, export_format)
            print(f"  [export] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cast-aware audiobook generation")
    parser.add_argument("--script", required=True, help="Script file or directory of .md files")
    parser.add_argument("--cast", required=True, help="cast.toml")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--chapter", help="Only generate chapters matching this substring")
    parser.add_argument("--export", default="wav", choices=["wav", "mp3", "flac", "m4a"])
    parser.add_argument("--no-stitch", action="store_true", help="Skip chapter stitching")
    parser.add_argument("--force", action="store_true", help="Regenerate even if WAV exists")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip loudness normalization at stitch time (voices will not match)",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=DEFAULT_TARGET_DBFS,
        help=f"Per-line RMS target in dBFS (default {DEFAULT_TARGET_DBFS}; ACX wants -23 to -18)",
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=DEFAULT_PEAK_DBFS,
        help=f"Peak ceiling in dBFS (default {DEFAULT_PEAK_DBFS}; ACX wants -3 or lower)",
    )
    parser.add_argument(
        "--restitch",
        action="store_true",
        help="Re-stitch existing line WAVs without generating anything new",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Generate this many lines concurrently, one model copy per worker "
            "(~4.6GB each). Default 1. On Apple Silicon 2 measured fastest; "
            "3+ contends for the GPU and gives back most of the gain."
        ),
    )
    args = parser.parse_args()

    print(f"Parsing script: {args.script}")
    lines = parse_script_path(args.script)
    if args.chapter:
        needle = args.chapter.lower()
        lines = [ln for ln in lines if needle in ln.chapter.lower()]
    if not lines:
        print("No script lines to generate.")
        sys.exit(1)

    print(f"Loading cast: {args.cast}")
    cast = load_cast(args.cast)

    # Validation
    seen = speakers_in(lines)
    chaps = chapters_in(lines)
    unmapped = validate_cast_against_script(cast, seen)
    if unmapped:
        print(f"WARNING: unmapped speakers (will fall back to '{cast.default}'): {unmapped}")
    present, missing = validate_cast_against_voices(cast)
    if missing:
        print(f"ERROR: missing voices in library: {missing}")
        print("Run `python -m audiobook.bootstrap --voices voices.toml` first.")
        sys.exit(2)

    print(f"Script: {len(lines)} lines across {len(chaps)} chapter(s), {len(seen)} speaker(s)")
    for ch in chaps:
        n = sum(1 for ln in lines if ln.chapter == ch)
        print(f"  - {ch}: {n} lines")

    os.makedirs(args.out, exist_ok=True)
    if args.restitch:
        print("Re-stitch only: using existing line WAVs, generating nothing.")
        by_chapter = plan_paths(lines, args.out)
    elif args.workers > 1:
        by_chapter = generate_lines_parallel(
            lines, args.cast, args.out, workers=args.workers, force=args.force
        )
    else:
        by_chapter = generate_lines(lines, cast, args.out, force=args.force)

    if not args.no_stitch:
        print("\nStitching chapters...")
        stitch_chapters(
            by_chapter,
            args.out,
            export_format=args.export,
            normalize=not args.no_normalize,
            target_dbfs=args.target_dbfs,
            peak_dbfs=args.peak_dbfs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
