import copy
import json

import pytest

from pipeline import minutes as M
from pipeline.errors import PipelineError

TRANSCRIPT = (
    "Let's talk about the website. The checkout page is slow. Maybe we should hire a contractor, "
    "but we haven't agreed. We agreed to launch the new logo on 3 June. Dana will fix the checkout "
    "bug by Friday. Somebody should look at the hosting costs. Marcus said he will write the runbook."
)

GOOD = {
    "summary": "The team reviewed the website.",
    "minutes": [{"topic": "Website", "points": ["The checkout page is slow."]}],
    "decisions": [{"decision": "Launch the new logo on 3 June.",
                   "evidence": "We agreed to launch the new logo on 3 June."}],
    "proposals": [{"proposal": "Hire a contractor.",
                   "evidence": "Maybe we should hire a contractor, but we haven't agreed."}],
    "action_items": [
        {"task": "Fix the checkout bug.", "owner": "Dana", "deadline": "Friday",
         "evidence": "Dana will fix the checkout bug by Friday."},
        {"task": "Write the runbook.", "owner": "Marcus", "deadline": "Unspecified",
         "evidence": "Marcus said he will write the runbook."},
    ],
}


class FakeGroq:
    """Stands in for the Groq client: returns the queued answers in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sent = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kwargs):
        self.sent.append(list(messages))

        class Msg:
            pass

        class Choice:
            pass

        class R:
            pass
        m = Msg()
        m.content = self.answers.pop(0)
        c = Choice()
        c.message = m
        r = R()
        r.choices = [c]
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)


def js(d):
    return json.dumps(d)


def test_valid_answer_is_accepted_unchanged():
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(GOOD)]))
    assert res.record["decisions"] == GOOD["decisions"]
    assert res.record["action_items"][0]["owner"] == "Dana"
    assert res.attempts == 1 and res.warnings == []


def test_invalid_json_is_retried_then_accepted():
    client = FakeGroq(["this is not json", js(GOOD)])
    res = M.generate_minutes(TRANSCRIPT, client=client)
    assert res.attempts == 2
    assert "not valid JSON" in client.sent[1][-1]["content"]


def test_wrong_shape_is_retried_with_reason():
    bad = copy.deepcopy(GOOD)
    del bad["proposals"]
    client = FakeGroq([js(bad), js(GOOD)])
    M.generate_minutes(TRANSCRIPT, client=client)
    assert "'proposals' must be a list" in client.sent[1][-1]["content"]


def test_gives_up_after_two_retries_with_clear_error():
    client = FakeGroq(["nope", "still nope", "no again"])
    with pytest.raises(PipelineError, match="could not be generated"):
        M.generate_minutes(TRANSCRIPT, client=client)
    assert len(client.sent) == 3   # first try + 2 retries, no more


def test_json_in_code_fence_is_accepted():
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq(["```json\n" + js(GOOD) + "\n```"]))
    assert res.record["summary"]


def test_reasoning_think_block_is_ignored():
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq(["<think>hmm {not json}</think>" + js(GOOD)]))
    assert res.record["summary"] and res.attempts == 1


def test_invented_owner_and_deadline_are_reset():
    bad = copy.deepcopy(GOOD)
    bad["action_items"][0]["owner"] = "Priya"            # never mentioned
    bad["action_items"][0]["deadline"] = "2026-10-09"    # invented date
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(bad)]))
    item = res.record["action_items"][0]
    assert item["owner"] == "Unspecified" and item["deadline"] == "Unspecified"
    assert len([w for w in res.warnings if "reset" in w]) == 2


def test_stated_owner_and_deadline_survive():
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(GOOD)]))
    assert res.record["action_items"][0]["deadline"] == "Friday"
    assert res.record["action_items"][1]["owner"] == "Marcus"


def test_pronoun_owner_is_reset():
    bad = copy.deepcopy(GOOD)
    bad["action_items"][1]["owner"] = "I"
    res = M.generate_minutes("I will write the runbook. " + TRANSCRIPT, client=FakeGroq([js(bad)]))
    assert res.record["action_items"][1]["owner"] == "Unspecified"


def test_empty_or_missing_owner_becomes_unspecified():
    bad = copy.deepcopy(GOOD)
    bad["action_items"][0]["owner"] = ""
    bad["action_items"][0]["deadline"] = "unspecified"
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(bad)]))
    assert res.record["action_items"][0]["owner"] == "Unspecified"
    assert res.record["action_items"][0]["deadline"] == "Unspecified"


def test_fabricated_quote_gives_warning():
    bad = copy.deepcopy(GOOD)
    bad["decisions"][0]["evidence"] = "The board approved a ten million dollar budget."
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(bad)]))
    assert any("not found word-for-word" in w for w in res.warnings)


def test_hedged_decision_gives_warning():
    bad = copy.deepcopy(GOOD)
    bad["decisions"].append({"decision": "Hire a contractor.",
                             "evidence": "Maybe we should hire a contractor, but we haven't agreed."})
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(bad)]))
    assert any("may be a proposal" in w for w in res.warnings)


def test_empty_lists_are_valid():
    empty = {"summary": "Nothing decided.", "minutes": [], "decisions": [], "proposals": [], "action_items": []}
    res = M.generate_minutes(TRANSCRIPT, client=FakeGroq([js(empty)]))
    assert res.record["decisions"] == [] and res.record["action_items"] == []


def test_empty_transcript_is_rejected():
    with pytest.raises(PipelineError, match="no transcript"):
        M.generate_minutes("   ", client=FakeGroq([]))


# ------------------------------------------------------- long transcripts
def _two_parts(monkeypatch):
    monkeypatch.setattr(M, "CHUNK_WORDS", 30)
    part1 = {"summary": "S1.", "minutes": [{"topic": "A", "points": ["a"]}],
             "decisions": [{"decision": "Launch the logo.", "evidence": "We agreed to launch the new logo on 3 June."}],
             "proposals": [], "action_items": []}
    part2 = {"summary": "S2.", "minutes": [{"topic": "B", "points": ["b"]}],
             "decisions": [{"decision": "launch the logo", "evidence": "We agreed to launch the new logo on 3 June."}],
             "proposals": [],
             "action_items": [{"task": "Write the runbook.", "owner": "Marcus", "deadline": "Unspecified",
                               "evidence": "Marcus said he will write the runbook."}]}
    return part1, part2


def test_long_transcript_is_split_and_merged(monkeypatch):
    part1, part2 = _two_parts(monkeypatch)
    client = FakeGroq([js(part1), js(part2), "Combined summary."])
    res = M.generate_minutes(TRANSCRIPT, client=client)
    assert res.parts == 2
    assert [t["topic"] for t in res.record["minutes"]] == ["A", "B"]      # order kept
    assert len(res.record["decisions"]) == 1                                # duplicate dropped
    assert res.record["action_items"][0]["owner"] == "Marcus"
    assert res.record["summary"] == "Combined summary."


def test_summary_falls_back_to_joined_parts_if_final_call_fails(monkeypatch):
    part1, part2 = _two_parts(monkeypatch)

    class Failing(FakeGroq):
        def create(self, model, messages, **kwargs):
            if not self.answers:
                raise RuntimeError("rate limited")
            return super().create(model, messages, **kwargs)
    res = M.generate_minutes(TRANSCRIPT, client=Failing([js(part1), js(part2)]))
    assert res.record["summary"] == "S1. S2."
