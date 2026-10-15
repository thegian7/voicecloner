"""ACX-ready export: every exported chapter must pass Audible's checks."""

import os

import numpy as np
import soundfile as sf

from core.acx import (
    PEAK_MAX_DBFS, acx_check, check_file, export_acx, lin_to_db, master_for_acx, soft_limit,
)


def speechlike(sr=48000, seconds=6.0, level=0.3, seed=0, stereo=False):
    """Bursts of harmonic 'speech' with pauses between them (TTS output is like this)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    voice = level * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.1 * rng.normal(size=t.size))
    gate = (np.sin(2 * np.pi * 0.7 * t) > -0.3).astype(float)
    x = voice * gate
    return np.stack([x, x * 0.9], axis=1) if stereo else x


def test_quiet_stereo_48k_tts_is_mastered_into_spec():
    out = master_for_acx(speechlike(level=0.01, stereo=True), 48000)
    r = acx_check(out, 44100)
    assert r.passes, r.summary()
    assert -23 <= r.rms_dbfs <= -18 and r.peak_dbfs <= PEAK_MAX_DBFS


def test_hot_peaky_audio_is_limited_under_the_ceiling():
    x = speechlike(level=0.2)
    x[1000:1010] = 0.99  # transient spikes
    out = master_for_acx(x, 48000)
    r = acx_check(out, 44100)
    assert r.passes, r.summary()


def test_room_tone_is_quiet_noise_not_digital_silence():
    out = master_for_acx(speechlike(), 48000)
    head = out[: int(0.5 * 44100)]
    assert np.any(head != 0.0)
    assert lin_to_db(float(np.sqrt(np.mean(head**2)))) < -60


def test_check_flags_each_requirement():
    sr = 44100
    loud = np.concatenate([np.zeros(sr), 0.9 * np.ones(sr), np.zeros(sr * 6)])
    r = acx_check(loud, 22050, channels=2)
    text = " ".join(r.failures)
    for needle in ["sample rate", "channels", "RMS", "peak", "head room tone", "tail room tone"]:
        assert needle in text, (needle, text)


def test_soft_limit_never_exceeds_the_ceiling_and_is_transparent_below_the_knee():
    x = np.linspace(-1.5, 1.5, 10001)
    y = soft_limit(x, ceiling_db=-3.6)
    assert np.abs(y).max() <= 10 ** (-3.6 / 20) + 1e-9
    small = np.linspace(-0.1, 0.1, 101)
    assert np.allclose(soft_limit(small), small)


def test_exported_mp3_is_acx_compliant_after_decoding(tmp_path):
    wav = tmp_path / "chapter.wav"
    sf.write(wav, speechlike(level=0.05, stereo=True), 48000)
    mp3 = tmp_path / "chapter.mp3"
    report = export_acx(str(wav), str(mp3))
    assert os.path.exists(mp3)
    assert report.passes, report.summary()
    assert report.sample_rate == 44100 and report.channels == 1
    assert report.bitrate_kbps == 192 and report.cbr
    # The temp mastering file is cleaned up.
    assert not any(p.name.endswith(".acx.wav") for p in tmp_path.iterdir())
    # And an independent check of the written file agrees.
    assert check_file(str(mp3)).passes


def test_resampling_keeps_duration_and_pitch():
    # 6 s at 48 kHz must still be 6 s at 44.1 kHz (plus room tone), or the narration plays slow and low.
    out = master_for_acx(speechlike(sr=48000, seconds=6.0), 48000)
    assert abs(len(out) / 44100 - (6.0 + 0.75 + 2.5)) < 0.01


def test_audiobook_acx_export_writes_one_checked_mp3_per_chapter(tmp_path, monkeypatch):
    import audiobook.export as export_mod
    from audiobook.state import BookState, ChunkState

    monkeypatch.setattr(export_mod, "TEMP_DIR", str(tmp_path))
    chunks = []
    for ch in ["Chapter 1 - A Scandal in Bohemia", "Chapter 2 - The Red-Headed League"]:
        for i in range(2):
            p = tmp_path / f"{ch[:9]}-{i}.wav"
            sf.write(p, speechlike(seconds=3.0, level=0.02 + 0.03 * i, seed=i), 48000)
            chunks.append(ChunkState(text="x", chapter=ch, audio_path=str(p), approved=True, is_para_end=i == 1))
    chunks.append(ChunkState(text="draft", chapter="Chapter 3", audio_path=None, approved=False))
    paths, msg = export_mod.export_audiobook(BookState(chunks=chunks), fmt="acx")
    assert len(paths) == 2 and all(p.endswith(".mp3") for p in paths)
    assert "2 pass Audible's checks" in msg, msg
    assert all(check_file(p).passes for p in paths)
