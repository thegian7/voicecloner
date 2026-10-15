"""ACX (Audible) mastering and checks: "ACX-ready export".

ACX's audio submission requirements, which every exported chapter must meet:

- each file is one chapter, 44.1 kHz, mono, MP3 at 192 kbps or higher, CBR
- RMS between -23 and -18 dBFS
- peak at or below -3 dBFS
- noise floor at or below -60 dBFS RMS
- 0.5-1 s of room tone at the start and 1-5 s at the end

TTS output is clean, so the work here is level (RMS into range, peaks under
the ceiling with a soft limiter), format, and room tone. Room tone is very
quiet noise rather than digital silence, which ACX reviewers flag.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np
import soundfile as sf

ACX_SAMPLE_RATE = 44100
ACX_BITRATE = "192k"
RMS_MIN_DBFS, RMS_MAX_DBFS = -23.0, -18.0
TARGET_RMS_DBFS = -20.0
PEAK_MAX_DBFS = -3.0
NOISE_FLOOR_MAX_DBFS = -60.0
HEAD_TONE_S = 0.75
TAIL_TONE_S = 2.5
ROOM_TONE_DBFS = -78.0
# Margin under the ceiling so MP3 encoding (which can add inter-sample overs)
# still measures below -3 dBFS after decoding.
LIMIT_DBFS = PEAK_MAX_DBFS - 0.6


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x: np.ndarray, sr: int, target: int = ACX_SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return x
    from scipy.signal import resample_poly

    g = int(np.gcd(sr, target))
    return resample_poly(x, target // g, sr // g)


def soft_limit(x: np.ndarray, ceiling_db: float = LIMIT_DBFS, knee_db: float = 6.0) -> np.ndarray:
    """Transparent below the knee; above it, samples curve smoothly toward the ceiling, never past it."""
    ceiling = db_to_lin(ceiling_db)
    knee = db_to_lin(ceiling_db - knee_db)
    a = np.abs(x)
    over = a > knee
    if not np.any(over):
        return x
    span = ceiling - knee
    y = a.copy()
    y[over] = knee + span * np.tanh((a[over] - knee) / span)
    return np.sign(x) * y


def room_tone(seconds: float, sr: int, level_dbfs: float = ROOM_TONE_DBFS, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, db_to_lin(level_dbfs), int(round(seconds * sr)))


def master_for_acx(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return ACX-mastered mono float audio at 44.1 kHz, room tone included."""
    x = resample(to_mono(audio), sr)
    if x.size == 0 or not np.any(x):
        raise ValueError("cannot master silence")
    rms = float(np.sqrt(np.mean(x**2)))
    x = x * (db_to_lin(TARGET_RMS_DBFS) / rms)
    # The limiter only touches peaks, so RMS barely moves; correct the small
    # loss once, then limit again so the ceiling always holds.
    x = soft_limit(x)
    rms = float(np.sqrt(np.mean(x**2)))
    if lin_to_db(rms) < TARGET_RMS_DBFS - 0.5:
        x = soft_limit(x * (db_to_lin(TARGET_RMS_DBFS) / rms))
    head = room_tone(HEAD_TONE_S, ACX_SAMPLE_RATE, seed=1)
    tail = room_tone(TAIL_TONE_S, ACX_SAMPLE_RATE, seed=2)
    return np.concatenate([head, x, tail])


@dataclass
class AcxReport:
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_dbfs: float
    noise_floor_dbfs: float
    head_tone_s: float
    tail_tone_s: float
    bitrate_kbps: int | None = None
    cbr: bool | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passes"] = self.passes
        return d

    def summary(self) -> str:
        status = "PASS" if self.passes else "FAIL: " + "; ".join(self.failures)
        return (
            f"ACX {status} · RMS {self.rms_dbfs:.1f} dBFS · peak {self.peak_dbfs:.1f} dBFS · "
            f"noise floor {self.noise_floor_dbfs:.1f} dBFS · room tone {self.head_tone_s:.2f}s/{self.tail_tone_s:.2f}s"
        )


def _tone_length(x: np.ndarray, sr: int, from_end: bool, gate_db: float = -55.0) -> float:
    gate = db_to_lin(gate_db)
    idx = np.flatnonzero(np.abs(x) > gate)
    if idx.size == 0:
        return len(x) / sr
    n = (len(x) - 1 - idx[-1]) if from_end else idx[0]
    return float(n) / sr


def acx_check(audio: np.ndarray, sr: int, channels: int = 1) -> AcxReport:
    """Measure audio against the ACX requirements (format checks are added by check_file)."""
    x = to_mono(audio)
    rms = lin_to_db(float(np.sqrt(np.mean(x**2)))) if x.size else -120.0
    peak = lin_to_db(float(np.abs(x).max())) if x.size else -120.0
    head = _tone_length(x, sr, from_end=False)
    tail = _tone_length(x, sr, from_end=True)
    floor_seg = x[: max(1, int(min(head, 0.5) * sr))]
    noise = lin_to_db(float(np.sqrt(np.mean(floor_seg**2)))) if floor_seg.size else -120.0
    r = AcxReport(sr, channels, round(rms, 2), round(peak, 2), round(noise, 2), round(head, 2), round(tail, 2))
    if sr != ACX_SAMPLE_RATE:
        r.failures.append(f"sample rate {sr} Hz (needs 44100)")
    if channels != 1:
        r.failures.append(f"{channels} channels (needs mono)")
    if not RMS_MIN_DBFS <= rms <= RMS_MAX_DBFS:
        r.failures.append(f"RMS {rms:.1f} dBFS outside -23..-18")
    if peak > PEAK_MAX_DBFS:
        r.failures.append(f"peak {peak:.1f} dBFS above -3")
    if noise > NOISE_FLOOR_MAX_DBFS:
        r.failures.append(f"noise floor {noise:.1f} dBFS above -60")
    if not 0.5 <= head <= 1.0:
        r.failures.append(f"head room tone {head:.2f}s (needs 0.5-1)")
    if not 1.0 <= tail <= 5.0:
        r.failures.append(f"tail room tone {tail:.2f}s (needs 1-5)")
    return r


def _ffprobe_bitrate(path: str) -> tuple[int | None, bool | None]:
    if not shutil.which("ffprobe"):
        return None, None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate,codec_name",
         "-of", "default=nw=1", path],
        capture_output=True, text=True, check=False,
    ).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    try:
        kbps = int(fields.get("bit_rate", "0")) // 1000
    except ValueError:
        kbps = None
    # libmp3lame with -b:a is CBR; ffprobe reports the nominal rate exactly for CBR.
    return kbps, (kbps is not None and kbps >= 192 and fields.get("codec_name") == "mp3")


def check_file(path: str) -> AcxReport:
    """Decode an exported file and check it, including bitrate/CBR for MP3s."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(path)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64) / float(1 << (8 * seg.sample_width - 1))
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels)
    report = acx_check(samples, seg.frame_rate, seg.channels)
    if path.lower().endswith(".mp3"):
        report.bitrate_kbps, report.cbr = _ffprobe_bitrate(path)
        if not report.cbr:
            report.failures.append(f"MP3 bitrate {report.bitrate_kbps} kbps / not CBR 192+")
    else:
        report.failures.append("not an MP3 (ACX needs MP3 192 kbps CBR)")
    return report


def export_acx(input_wav: str, output_mp3: str) -> AcxReport:
    """Master a stitched chapter WAV for ACX, write a 192 kbps CBR mono MP3, and check the result."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for ACX export (brew install ffmpeg / apt install ffmpeg)")
    audio, sr = sf.read(input_wav, always_2d=False)
    mastered = master_for_acx(audio, sr)
    tmp_wav = output_mp3 + ".acx.wav"
    sf.write(tmp_wav, mastered.astype(np.float32), ACX_SAMPLE_RATE, subtype="FLOAT")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav, "-ac", "1", "-ar", str(ACX_SAMPLE_RATE),
             "-codec:a", "libmp3lame", "-b:a", ACX_BITRATE, output_mp3],
            check=True,
        )
    finally:
        try:
            import os

            os.remove(tmp_wav)
        except OSError:
            pass
    return check_file(output_mp3)
