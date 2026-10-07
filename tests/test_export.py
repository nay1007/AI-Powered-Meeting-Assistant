import json
import re

from pipeline.export import FILES, save_outputs, to_json, to_markdown
from pipeline.schema import MeetingRecord


def make(**over):
    base = dict(
        raw_transcript="we use cooper netties. Dana will fix it by Friday.",
        refined_transcript="we use Kubernetes. Dana will fix it by Friday.",
        summary="A short meeting.",
        minutes=[{"topic": "Infra", "points": ["Kubernetes is used."]}],
        decisions=[{"decision": "Use Kubernetes.", "evidence": "we use Kubernetes"}],
        proposals=[{"proposal": "Hire help.", "evidence": "maybe hire"}],
        action_items=[{"task": "Fix it.", "owner": "Dana", "deadline": "Friday", "evidence": "Dana will fix it"},
                      {"task": "Review costs.", "owner": "Unspecified", "deadline": "Unspecified", "evidence": "costs"}],
        warnings=["something odd"],
        metadata={"source_file": "a.wav"},
    )
    base.update(over)
    return MeetingRecord(**base)


def test_markdown_and_json_contain_the_same_items():
    rec = make()
    md, data = to_markdown(rec), json.loads(to_json(rec))
    for key, field in (("decisions", "decision"), ("proposals", "proposal"), ("action_items", "task")):
        for item in data[key]:
            assert item[field] in md
    for a in data["action_items"]:
        assert a["owner"] in md and a["deadline"] in md
    assert data["summary"] in md
    assert data["raw_transcript"] in md and data["refined_transcript"] in md
    assert data["warnings"][0] in md


def test_counts_match_between_formats():
    rec = make()
    md, data = to_markdown(rec), json.loads(to_json(rec))
    table_rows = [l for l in md.splitlines() if re.match(r"\| \d+ \|", l)]
    assert len(table_rows) == len(data["action_items"])
    assert len(re.findall(r"^\d+\. ", md, flags=re.M)) == len(data["decisions"]) + len(data["proposals"])


def test_empty_lists_exported_as_empty_and_none():
    rec = make(decisions=[], proposals=[], action_items=[], minutes=[], warnings=[])
    md, data = to_markdown(rec), json.loads(to_json(rec))
    assert data["decisions"] == [] and data["action_items"] == []
    assert md.count("None.") >= 5


def test_save_outputs_writes_every_listed_file(tmp_path):
    paths = save_outputs(make(), str(tmp_path / "out"))
    assert set(paths) == set(FILES)
    assert (tmp_path / "out" / "raw_transcript.txt").read_text(encoding="utf-8").startswith("we use cooper")
    assert json.loads((tmp_path / "out" / "meeting_record.json").read_text(encoding="utf-8"))["summary"]
    assert (tmp_path / "out" / "meeting_record.md").read_text(encoding="utf-8").startswith("# Meeting Record")
