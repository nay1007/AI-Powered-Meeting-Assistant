import pytest

from pipeline import refine as R
from pipeline.checks import find_problems, negations_in, numbers_in


class FakeGroq:
    """Returns scripted answers and records the messages it was sent."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sent = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, **extra):
        self.sent.append(list(messages))

        class Msg:  # mimics the real response shape
            pass
        m = Msg()
        m.content = self.answers.pop(0)
        r = Msg()
        r.choices = [Msg()]
        r.choices[0].message = m
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)


RAW = "We will deploy on cooper net ease by 15 March. We do not need a rollback plan."


def test_good_fix_passes_first_time():
    fixed = "We will deploy on Kubernetes by 15 March. We do not need a rollback plan."
    client = FakeGroq([fixed])
    res = R.refine(RAW, client=client)
    assert res.refined == fixed
    assert res.chunks[0].status == "ok"


def test_changed_number_is_caught_and_retried_with_reason():
    bad = "We will deploy on Kubernetes by 50 March. We do not need a rollback plan."
    good = "We will deploy on Kubernetes by 15 March. We do not need a rollback plan."
    client = FakeGroq([bad, good])
    res = R.refine(RAW, client=client)
    assert res.refined == good
    assert res.chunks[0].status == "retried" and res.chunks[0].attempts == 2
    retry_message = client.sent[1][-1]["content"]
    assert "Numbers changed" in retry_message and "'15'" in retry_message


def test_dropped_negation_is_caught():
    bad = "We will deploy on Kubernetes by 15 March. We do need a rollback plan."
    assert any("Negations changed" in p for p in find_problems(RAW, bad))


def test_three_failures_fall_back_to_raw():
    bad = "We will deploy on Kubernetes by 50 March. We do not need a rollback plan."
    client = FakeGroq([bad, bad, bad])
    res = R.refine(RAW, client=client)
    assert res.refined == RAW and res.chunks[0].status == "fell_back_to_raw"
    assert res.chunks[0].problems


def test_rewriting_or_deleting_text_is_caught():
    assert any("too large" in p for p in find_problems(RAW, "We will deploy."))
    # a whole sentence silently dropped from a longer text
    long_raw = " ".join(f"Item {i} was reviewed today." for i in range(20))
    assert find_problems(long_raw, long_raw.replace("Item 7 was reviewed today. ", ""))


def test_merging_spaced_letters_is_allowed():
    assert not find_problems("We move to g R P C soon.", "We move to gRPC soon.")


def test_number_words_and_contractions_are_counted():
    assert numbers_in("fifteen and 1,200 and 3.5")["fifteen"] == 1
    assert numbers_in("1,200")["1200"] == 1
    assert negations_in("we don't, can't and never")["n't"] == 2


def test_highlighting_marks_changes():
    out = R.highlighted_html("deploy on cooper net ease now", "deploy on Kubernetes now")
    assert 'title="Original: cooper net ease"' in out and ">Kubernetes</mark>" in out



def test_chunking_keeps_all_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = R.split_into_chunks(text, max_words=100)
    assert len(chunks) > 1 and " ".join(chunks) == text
