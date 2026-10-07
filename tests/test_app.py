import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

RECORD = {
    "metadata": {"duration_seconds": 47.8, "refinement_parts_fell_back_to_raw": 0},
    "raw_transcript": "We run on cooper net ease.",
    "refined_transcript": "We run on Kubernetes.",
    "summary": "A short sync.",
    "minutes": [{"topic": "Deploy", "points": ["Moving to Kubernetes."]}],
    "decisions": [{"decision": "Ship in March.", "evidence": "we ship in March"}],
    "proposals": [{"proposal": "Use Grafana.", "evidence": "maybe Grafana"}],
    "action_items": [{"task": "Write runbook.", "owner": "Marcus", "deadline": "Unspecified", "evidence": "Marcus will"}],
    "warnings": ["Example warning."],
}
FILES = {n: "x" for n in ("meeting_record.md", "meeting_record.json", "raw_transcript.txt", "refined_transcript.txt")}


def fresh():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_first_screen_renders_and_button_is_disabled():
    at = fresh()
    assert not at.exception
    assert at.title[0].value.endswith("Meeting Assistant")
    assert at.button[0].disabled          # nothing uploaded yet
    assert not at.tabs                    # no results yet


def test_results_are_shown_in_every_tab():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["result"] = {"record": RECORD, "files": FILES}
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Summary", "Transcripts", "Minutes", "Decisions", "Action items", "Warnings", "Downloads"]
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Ship in March." in text and "Use Grafana." in text
    assert 'title="Original: cooper net ease."' in text and ">Kubernetes.</mark>" in text   # highlighted in place
    assert any("Example warning." in w.value for w in at.warning)


def test_error_message_is_shown_not_a_crash():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["error"] = "The file is empty (0 bytes)."
    at.run()
    assert not at.exception
    assert at.error[0].value == "The file is empty (0 bytes)."
