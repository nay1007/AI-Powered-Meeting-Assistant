import pytest

from pipeline.errors import PipelineError
from pipeline.transcribe import transcribe
from tests.test_validate import make_wav


class FakeClient:
    """Stands in for Groq so tests need no internet or key."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.audio = self
        self.transcriptions = self

    def create(self, **kwargs):
        assert kwargs["model"] == "whisper-large-v3"
        assert kwargs["language"] == "en"
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_cut_uses_longest_pause_in_window():
    from pipeline.transcribe import choose_cuts
    silences = [(590.0, 590.5), (605.0, 607.0), (700.0, 702.0)]  # 700 is outside +/-30 s
    cuts = choose_cuts(1500, silences)
    assert cuts[0] == (606.0, False)    # middle of the 2 s pause near 600


def test_no_pause_cuts_at_target_with_overlap():
    from pipeline.transcribe import choose_cuts
    assert choose_cuts(1500, []) == [(600.0, True), (1200.0, True)]


def test_short_recording_is_one_file(tmp_path):
    from pipeline.transcribe import split_audio
    p = tmp_path / "a.wav"
    make_wav(p, seconds=2)
    files, flags = split_audio(str(p), str(tmp_path))
    assert len(files) == 1 and flags == []


def test_merge_removes_overlap_only_when_flagged():
    from pipeline.transcribe import merge_transcripts
    a = "we agreed to ship on the fifteenth of march and then"
    b = "of March and then review the budget"
    assert merge_transcripts([a, b], [True]) == (
        "we agreed to ship on the fifteenth of march and then review the budget")
    # clean silence cut: nothing is removed even if words repeat
    assert merge_transcripts(["go on and on", "on and on again"], [False]) == "go on and on on and on again"


def test_joins_chunks_and_returns_text(tmp_path):
    p = tmp_path / "a.wav"
    make_wav(p, seconds=2)
    client = FakeClient(["hello team"])
    assert transcribe(str(p), client=client) == "hello team"


def test_empty_result_gives_clear_error(tmp_path):
    p = tmp_path / "a.wav"
    make_wav(p, seconds=2)
    with pytest.raises(PipelineError, match="No speech"):
        transcribe(str(p), client=FakeClient(["   "]))


def test_api_failure_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.transcribe.time.sleep", lambda s: None)
    p = tmp_path / "a.wav"
    make_wav(p, seconds=2)
    client = FakeClient([RuntimeError("boom")] * 3)
    with pytest.raises(PipelineError, match="failed after 3 tries"):
        transcribe(str(p), client=client)
    assert client.calls == 3
