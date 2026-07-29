import shutil
import subprocess

import numpy as np
import soundfile as sf


def crossfade(a: np.ndarray, b: np.ndarray, overlap_samples: int) -> np.ndarray:
    """Crossfade two audio arrays over overlap_samples."""
    if overlap_samples <= 0 or len(a) < overlap_samples or len(b) < overlap_samples:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=a.dtype)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=b.dtype)
    blended = a[-overlap_samples:] * fade_out + b[:overlap_samples] * fade_in
    return np.concatenate([a[:-overlap_samples], blended, b[overlap_samples:]])


def stitch_chunks(
    audio_list: list[np.ndarray],
    sr: int,
    pause_ms: int = 250,
    paragraph_pause_ms: int = 500,
    crossfade_ms: int = 40,
    paragraph_flags: list[bool] | None = None,
) -> np.ndarray:
    """Stitch audio chunks with silence gaps and crossfades.

    paragraph_flags: if provided, True means the chunk ends a paragraph (gets longer pause).
    """
    if not audio_list:
        return np.array([], dtype=np.float32)

    overlap = int(sr * crossfade_ms / 1000)
    # Collect all segments, then concatenate once (O(n) instead of O(n²))
    parts: list[np.ndarray] = []
    prev_tail = np.array([], dtype=np.float32)

    for i, chunk in enumerate(audio_list):
        if i == 0:
            if overlap > 0 and len(chunk) > overlap and len(audio_list) > 1:
                parts.append(chunk[:-overlap])
                prev_tail = chunk[-overlap:]
            else:
                parts.append(chunk)
            continue

        is_para = paragraph_flags[i - 1] if paragraph_flags else False
        gap = paragraph_pause_ms if is_para else pause_ms
        silence = np.zeros(int(sr * gap / 1000), dtype=np.float32)

        if overlap > 0 and len(prev_tail) == overlap and len(chunk) >= overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            blended = prev_tail * fade_out + chunk[:overlap] * fade_in
            parts.append(silence)
            parts.append(blended)
            body = chunk[overlap:]
        else:
            parts.append(prev_tail)
            parts.append(silence)
            body = chunk

        if i < len(audio_list) - 1 and overlap > 0 and len(body) > overlap:
            parts.append(body[:-overlap])
            prev_tail = body[-overlap:]
        else:
            parts.append(body)
            prev_tail = np.array([], dtype=np.float32)

    if len(prev_tail) > 0:
        parts.append(prev_tail)

    return np.concatenate(parts)


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr)


# --- Loudness ---------------------------------------------------------------
#
# Voice-design generates each voice's reference sample independently, at
# whatever level the model happens to produce, and cloning inherits that level.
# Nothing downstream corrects for it, so a multi-voice stitch can land wildly
# uneven: in THE LATTICE ch 1 the narrator sat ~16 dB under the dialogue, which
# means you set your volume for narration and get shouted at by every reply.
#
# ACX (Audible) spec, for reference: RMS between -23 and -18 dBFS, peak at or
# below -3 dBFS, noise floor at or below -60 dBFS.

DEFAULT_TARGET_DBFS = -20.0
DEFAULT_PEAK_DBFS = -3.0


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def speech_rms(audio: np.ndarray, gate_db: float = -40.0) -> float:
    """RMS over speech only, ignoring near-silence.

    A plain full-signal RMS punishes lines that carry long pauses — the narrator
    lines here are full of them — so those lines get over-boosted to compensate
    for their own silence. Gate relative to the line's own peak instead, so the
    measurement tracks how loud the voice actually is.
    """
    x = _mono(np.asarray(audio, dtype=np.float64))
    if x.size == 0:
        return 0.0
    peak = float(np.abs(x).max())
    if peak <= 0.0:
        return 0.0
    active = x[np.abs(x) > peak * _db_to_lin(gate_db)]
    if active.size == 0:
        active = x
    return float(np.sqrt(np.mean(active**2)))


def line_gain(
    audio: np.ndarray,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    peak_dbfs: float = DEFAULT_PEAK_DBFS,
    max_boost_db: float = 24.0,
) -> float:
    """Gain that brings one line to target RMS without pushing its peak past the ceiling.

    The peak guard wins over the RMS target: a line that would clip gets less
    gain rather than a limiter, which keeps this transparent and reversible.
    """
    x = _mono(np.asarray(audio, dtype=np.float64))
    rms = speech_rms(x)
    if rms <= 0.0:
        return 1.0

    gain = _db_to_lin(target_dbfs) / rms
    gain = min(gain, _db_to_lin(max_boost_db))

    peak = float(np.abs(x).max())
    if peak > 0.0:
        gain = min(gain, _db_to_lin(peak_dbfs) / peak)
    return float(gain)


def plan_gains(
    audio_paths: list[str],
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    peak_dbfs: float = DEFAULT_PEAK_DBFS,
) -> list[float]:
    """Per-line gains for a set of files, trimmed so the stitched result honours peak_dbfs.

    Two passes are needed to normalise without holding the whole chapter in
    memory. This is the cheap analysis pass: it reads each file, works out its
    gain, and predicts the post-gain peak. Crossfaded joins are a convex blend
    of two lines, so they can't exceed the louder one — that makes the global
    peak exactly max(peak_i * gain_i), computable without any audio in hand.
    """
    gains: list[float] = []
    global_peak = 0.0
    for path in audio_paths:
        audio, _ = sf.read(path)
        g = line_gain(audio, target_dbfs=target_dbfs, peak_dbfs=peak_dbfs)
        gains.append(g)
        peak = float(np.abs(_mono(audio)).max()) if audio.size else 0.0
        global_peak = max(global_peak, peak * g)

    ceiling = _db_to_lin(peak_dbfs)
    if global_peak > ceiling:
        trim = ceiling / global_peak
        gains = [g * trim for g in gains]
    return gains


def export_audio(input_wav: str, output_path: str, fmt: str = "wav") -> str:
    """Convert a WAV file to the requested format using pydub/ffmpeg.

    Supported formats: wav, mp3, flac, m4a.
    Returns the output path.
    """
    fmt = fmt.lower()
    if fmt == "wav":
        if input_wav != output_path:
            shutil.copy2(input_wav, output_path)
        return output_path

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for MP3/FLAC/M4A export. "
            "Install it: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    codec_args: dict[str, list[str]] = {
        "mp3": ["-codec:a", "libmp3lame", "-b:a", "192k"],
        "flac": ["-codec:a", "flac"],
        "m4a": ["-codec:a", "aac", "-b:a", "192k"],
    }
    if fmt not in codec_args:
        raise ValueError(f"Unsupported format: {fmt}. Use wav, mp3, flac, or m4a.")

    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        *codec_args[fmt],
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def stitch_to_file(
    audio_paths: list[str],
    output_path: str,
    pause_ms: int = 250,
    paragraph_pause_ms: int = 500,
    crossfade_ms: int = 40,
    paragraph_flags: list[bool] | None = None,
    gains: list[float] | None = None,
) -> tuple[int, float]:
    """Stitch audio files to disk incrementally. Returns (sr, duration_seconds).

    Only holds ~2 chunks in memory at a time.

    gains: optional per-file linear gain, one per path — see plan_gains(). Applied
    as each file is read, so crossfades and tails inherit it for free.
    """
    if not audio_paths:
        sf.write(output_path, np.array([], dtype=np.float32), 24000)
        return 24000, 0.0

    if gains is not None and len(gains) != len(audio_paths):
        raise ValueError(
            f"gains has {len(gains)} entries for {len(audio_paths)} files — must match"
        )

    def _read(i: int):
        audio, sr_ = sf.read(audio_paths[i])
        if gains is not None and gains[i] != 1.0:
            audio = audio * gains[i]
        return audio, sr_

    first, sr = _read(0)
    overlap = int(sr * crossfade_ms / 1000)
    total_frames = 0

    with sf.SoundFile(output_path, mode="w", samplerate=sr, channels=1) as out:
        # Write first chunk minus its tail (reserve for crossfade with next)
        if len(audio_paths) == 1:
            out.write(first)
            return sr, len(first) / sr

        # Keep only the tail of the previous chunk for crossfading
        prev_tail = first[-(overlap):] if overlap > 0 else np.array([], dtype=first.dtype)
        out.write(first[:-overlap] if overlap > 0 else first)
        total_frames += len(first) - (overlap if overlap > 0 else 0)
        del first

        for i in range(1, len(audio_paths)):
            is_para = paragraph_flags[i - 1] if paragraph_flags else False
            gap = paragraph_pause_ms if is_para else pause_ms
            silence = np.zeros(int(sr * gap / 1000), dtype=np.float32)

            chunk, _ = _read(i)

            # Crossfade prev_tail with start of this chunk
            if overlap > 0 and len(prev_tail) == overlap and len(chunk) >= overlap:
                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended = prev_tail * fade_out + chunk[:overlap] * fade_in
                out.write(silence)
                out.write(blended)
                total_frames += len(silence) + len(blended)
                body = chunk[overlap:]
            else:
                out.write(prev_tail)
                out.write(silence)
                total_frames += len(prev_tail) + len(silence)
                body = chunk

            # If not last chunk, reserve tail for next crossfade
            if i < len(audio_paths) - 1 and overlap > 0 and len(body) > overlap:
                out.write(body[:-overlap])
                total_frames += len(body) - overlap
                prev_tail = body[-overlap:]
            else:
                out.write(body)
                total_frames += len(body)
                prev_tail = np.array([], dtype=np.float32)

            del chunk, body

    return sr, total_frames / sr
