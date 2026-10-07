import pytest

from pipeline import transcribe as T
from pipeline.errors import PipelineError
from pipeline.transcribe import transcribe
from tests.test_validate import make_wav


class Seg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """Stands in for the local Whisper model so tests need no model download."""

    def __init__(self, segments=None, error=None):
        self.segments = segments or []
        self.error = error
        self.kwargs = None
        self.seen_file = None

    def transcribe(self, audio, **kwargs):
        self.seen_file = audio
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return iter(self.segments), None


def wav(tmp_path):
    p = tmp_path / "a.wav"
    make_wav(p, seconds=2)
    return str(p)


def test_segments_are_joined_into_one_transcript(tmp_path):
    model = FakeModel([Seg(" hello team."), Seg(" we ship on Friday. ")])
    assert transcribe(wav(tmp_path), model=model) == "hello team. we ship on Friday."


def test_model_is_asked_for_english_and_converted_flac(tmp_path):
    model = FakeModel([Seg("hi")])
    transcribe(wav(tmp_path), model=model)
    assert model.kwargs["language"] == "en"
    assert model.seen_file.endswith(".flac")


def test_empty_result_gives_clear_error(tmp_path):
    with pytest.raises(PipelineError, match="No speech"):
        transcribe(wav(tmp_path), model=FakeModel([Seg("   ")]))


def test_model_failure_gives_clear_error(tmp_path):
    with pytest.raises(PipelineError, match="Speech-to-text failed: boom"):
        transcribe(wav(tmp_path), model=FakeModel(error=RuntimeError("boom")))


def test_unconvertible_audio_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(PipelineError, match="could not be converted"):
        transcribe(str(bad), model=FakeModel([Seg("x")]))


def test_model_load_failure_gives_clear_error(monkeypatch):
    import faster_whisper

    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(T, "_model", None)
    monkeypatch.setattr(faster_whisper, "WhisperModel", boom)
    with pytest.raises(PipelineError, match="could not be loaded"):
        T.load_model()
