"""Crash recovery and the audiobook fixes ported from the February audit."""

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def session(tmp_path, monkeypatch):
    from audiobook import session as s

    monkeypatch.setattr(s, "SESSION_DIR", str(tmp_path / "session"))
    monkeypatch.setattr(s, "_STATE", str(tmp_path / "session" / "session.json"))
    return s


def _wav(path):
    sf.write(path, np.zeros(4800, dtype=np.float32), 48000)
    return str(path)


def test_session_round_trip_and_missing_audio_goes_back_to_pending(session):
    a = _wav(session.chunk_path("chunk_0000.wav"))
    b = _wav(session.chunk_path("chunk_0001.wav"))
    chunks = [
        {"text": "One.", "chapter": "Ch 1", "is_para_end": False, "audio_path": a, "approved": True},
        {"text": "Two.", "chapter": "Ch 1", "is_para_end": True, "audio_path": b, "approved": True},
        {"text": "Three.", "chapter": "Ch 2", "is_para_end": False, "audio_path": None, "approved": False},
    ]
    session.save(chunks, {"Dr.": "Doctor"})

    import os
    os.remove(b)  # audio lost (disk cleanup, crash mid-write): must be regenerated, not exported
    loaded, pron = session.load()
    assert [c["audio_path"] for c in loaded] == [a, None, None]
    assert [c["approved"] for c in loaded] == [True, False, False]
    assert loaded[1]["is_para_end"] is True
    assert pron == {"Dr.": "Doctor"}
    assert "3 chunks across 2 chapter(s): 1 generated, 1 approved" in session.summary(loaded)


def test_new_book_replaces_the_session(session):
    _wav(session.chunk_path("chunk_0000.wav"))
    session.save([{"text": "x", "chapter": "c", "is_para_end": False, "audio_path": None, "approved": False}])
    session.start()
    assert session.load() is None
    import glob, os
    assert glob.glob(os.path.join(session.SESSION_DIR, "chunk_*.wav")) == []


def test_corrupt_session_file_is_ignored(session):
    import os
    os.makedirs(session.SESSION_DIR, exist_ok=True)
    with open(session._STATE, "w") as f:
        f.write("{not json")
    assert session.load() is None


def test_bookstate_save_and_resume(session):
    from audiobook.state import BookState, ChunkState

    p = _wav(session.chunk_path("chunk_0000.wav"))
    book = BookState(chunks=[ChunkState(text="Hi.", chapter="Ch 1", audio_path=p, approved=True)])
    book.pronunciation["GRANBOARD"] = "Gran Board"
    book.save()
    back = BookState.resume()
    assert back.chunks == book.chunks
    assert back.pronunciation["GRANBOARD"] == "Gran Board"


def test_chapter_labels_parse_by_leading_number(session):
    from audiobook.chapters import chunk_chapters
    from audiobook.state import BookState
    from core import Chapter

    chapters = [Chapter(title="1.5 Hours Later", text="First chapter text."),
                Chapter(title="Two", text="Second chapter text.")]
    book = chunk_chapters(chapters, ["2. Two"], BookState())
    assert {c.chapter for c in book.chunks} == {"Two"}
    assert session.load() is not None  # chunking starts and saves a session
    with pytest.raises(ValueError):
        chunk_chapters(chapters, ["7. Gone"], BookState())  # stale label after reloading text


def test_unreadable_files_raise_readable_errors(tmp_path):
    from core import preprocess_file

    bad = tmp_path / "latin1.txt"
    bad.write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ValueError, match="isn't UTF-8"):
        preprocess_file(str(bad))
    with pytest.raises(ValueError, match="File not found"):
        preprocess_file(str(tmp_path / "nope.txt"))
    fake = tmp_path / "broken.epub"
    fake.write_text("not a zip")
    with pytest.raises(ValueError, match="as an EPUB"):
        preprocess_file(str(fake))


def test_ui_bridge_turns_valueerror_into_gradio_error():
    import gradio as gr
    from audiobook.ui import _on_export
    from audiobook.state import BookState

    with pytest.raises(gr.Error, match="No approved chunks"):
        _on_export(BookState(), "wav", False)
