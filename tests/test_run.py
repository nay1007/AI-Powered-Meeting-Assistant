import json

import pytest

from pipeline import run as R
from pipeline.errors import PipelineError
from pipeline.validate import AudioInfo


class Refined:
    refined = "We agreed to ship on Friday."
    chunks = []


class Minutes:
    record = {"summary": "S.", "minutes": [], "decisions": [], "proposals": [], "action_items": []}
    warnings = []
    parts = 1


def patch_all(monkeypatch, order):
    monkeypatch.setattr(R, "validate_audio", lambda p: order.append("validate") or AudioInfo(p, 5.0, -10.0))
    monkeypatch.setattr(R, "transcribe", lambda p: order.append("transcribe") or "raw text")
    monkeypatch.setattr(R, "refine", lambda raw, client=None: order.append("refine") or Refined())
    monkeypatch.setattr(R, "generate_minutes", lambda t, client=None: order.append("minutes") or Minutes())


def test_stages_run_in_order_and_files_are_saved(monkeypatch, tmp_path):
    order = []
    patch_all(monkeypatch, order)
    status = []
    res = R.run_pipeline("meeting.wav", str(tmp_path), on_status=status.append)
    assert order == ["validate", "transcribe", "refine", "minutes"]
    assert len(status) >= 4
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "meeting_record.json", "meeting_record.md", "raw_transcript.txt", "refined_transcript.txt"]
    data = json.loads((tmp_path / "meeting_record.json").read_text(encoding="utf-8"))
    assert data["raw_transcript"] == "raw text" and data["metadata"]["source_file"] == "meeting.wav"


def test_pipeline_error_passes_through(monkeypatch, tmp_path):
    patch_all(monkeypatch, [])
    monkeypatch.setattr(R, "validate_audio", lambda p: (_ for _ in ()).throw(PipelineError("The file is empty (0 bytes).")))
    with pytest.raises(PipelineError, match="empty"):
        R.run_pipeline("x.wav", str(tmp_path))


def test_unexpected_error_becomes_pipeline_error(monkeypatch, tmp_path):
    patch_all(monkeypatch, [])
    monkeypatch.setattr(R, "transcribe", lambda p: 1 / 0)
    with pytest.raises(PipelineError, match="unexpected"):
        R.run_pipeline("x.wav", str(tmp_path))


def test_refinement_fallback_is_reported_as_warning(monkeypatch, tmp_path):
    patch_all(monkeypatch, [])
    from pipeline.refine import ChunkReport

    class Fell(Refined):
        chunks = [ChunkReport(0, "fell_back_to_raw", 3, ["number changed"])]
    monkeypatch.setattr(R, "refine", lambda raw, client=None: Fell())
    res = R.run_pipeline("x.wav", str(tmp_path))
    assert any("left as the raw transcript" in w for w in res.record.warnings)
