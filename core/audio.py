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
) -> tuple[int, float]:
    """Stitch audio files to disk incrementally. Returns (sr, duration_seconds).

    Only holds ~2 chunks in memory at a time.
    """
    if not audio_paths:
        sf.write(output_path, np.array([], dtype=np.float32), 24000)
        return 24000, 0.0

    first, sr = sf.read(audio_paths[0])
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

            chunk, _ = sf.read(audio_paths[i])

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
