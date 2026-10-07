import math
import struct
import wave

import pytest

from pipeline.errors import PipelineError
from pipeline.validate import validate_audio


def make_wav(path, seconds=2.0, amplitude=0.5, rate=16000):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(n)
        )
        w.writeframes(frames)


def test_good_file_passes(tmp_path):
    p = tmp_path / "ok.wav"
    make_wav(p)
    info = validate_audio(str(p))
    assert info.duration_seconds == pytest.approx(2.0, abs=0.2)


def test_missing_file():
    with pytest.raises(PipelineError, match="No file"):
        validate_audio("does_not_exist.wav")


def test_wrong_type(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(PipelineError, match="Unsupported file type"):
        validate_audio(str(p))


def test_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(PipelineError, match="empty"):
        validate_audio(str(p))


def test_corrupt_file(tmp_path):
    p = tmp_path / "bad.mp3"
    p.write_bytes(b"this is definitely not audio" * 100)
    with pytest.raises(PipelineError, match="could not be read"):
        validate_audio(str(p))


def test_silent_file(tmp_path):
    p = tmp_path / "silent.wav"
    make_wav(p, amplitude=0.0)
    with pytest.raises(PipelineError, match="silent"):
        validate_audio(str(p))
