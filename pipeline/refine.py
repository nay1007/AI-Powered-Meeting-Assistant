"""Stage 2: raw transcript -> refined transcript (LLM #1, gpt-oss-120b on Groq).

The model only fixes misheard technical terms (see prompts/refine_prompt.txt).
After every answer, plain code (checks.py) compares it with the input. If a
number, a negation or the length changed, the model is told exactly what went
wrong and asked again (up to MAX_ATTEMPTS). If it still fails, that chunk keeps
the original raw text, so a bad answer can never reach the final record.
"""
import difflib
import os
import re
import time
from dataclasses import dataclass, field

from .checks import find_problems
from .config import REFINE_MODEL, require_key
from .errors import PipelineError

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "refine_prompt.txt")
CHUNK_WORDS = 800   # keeps each call well inside the free token-per-minute limit
MAX_ATTEMPTS = 3    # first try + 2 corrected retries


@dataclass
class ChunkReport:
    index: int
    status: str                      # "ok", "retried", or "fell_back_to_raw"
    attempts: int
    problems: list[str] = field(default_factory=list)  # problems from the LAST failed attempt


@dataclass
class RefineResult:
    refined: str
    chunks: list[ChunkReport]

    @property
    def fell_back(self) -> bool:
        return any(c.status == "fell_back_to_raw" for c in self.chunks)


def load_prompt() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read().strip()


def split_into_chunks(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Group whole sentences into chunks of about max_words words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for s in sentences:
        n = len(s.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _clean(output: str) -> str:
    out = output.strip()
    # swap lookalike characters for plain ones so the comparison with the input stays fair
    out = out.replace("‑", "-").replace("‐", "-").replace(" ", " ")
    out = re.sub(r"^```\w*\n|\n```$", "", out).strip()
    out = re.sub(r"^</?transcript>|</?transcript>$", "", out).strip()
    return out


def _call_llm(client, messages: list[dict]) -> str:
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=REFINE_MODEL, messages=messages, temperature=0.0,
                reasoning_effort="low")
            return _clean(resp.choices[0].message.content or "")
        except Exception as exc:  # network, rate limit...
            last = exc
            time.sleep(5 * (attempt + 1))
    raise PipelineError(f"Transcript refinement failed after 3 tries: {last}")


def refine_chunk(client, system_prompt: str, chunk: str, index: int) -> tuple[str, ChunkReport]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<transcript>\n{chunk}\n</transcript>"},
    ]
    problems: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        answer = _call_llm(client, messages)
        problems = find_problems(chunk, answer) if answer else ["Your output was empty."]
        if not problems:
            return answer, ChunkReport(index, "ok" if attempt == 1 else "retried", attempt)
        # Tell the model exactly what it got wrong and ask again.
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                "Your output was REJECTED by an automatic check:\n- " + "\n- ".join(problems) +
                "\nRedo the task from the original transcript, following every rule. "
                "Keep all numbers and negations exactly as in the original. "
                "If unsure about a change, leave the text unchanged. Output only the corrected transcript."},
        ]
    return chunk, ChunkReport(index, "fell_back_to_raw", MAX_ATTEMPTS, problems)


def refine(raw_transcript: str, client=None) -> RefineResult:
    if not raw_transcript.strip():
        raise PipelineError("There is no transcript to refine.")
    if client is None:
        from groq import Groq
        client = Groq(api_key=require_key("GROQ_API_KEY"))

    prompt = load_prompt()
    pieces, reports = [], []
    for i, chunk in enumerate(split_into_chunks(raw_transcript)):
        text, report = refine_chunk(client, prompt, chunk, i)
        pieces.append(text)
        reports.append(report)
    return RefineResult(" ".join(pieces), reports)


def word_changes(raw: str, refined: str) -> list[tuple[str, str, str]]:
    """Every change as (words_before, words_after, context). Word-level diff."""
    a, b = raw.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        context = " ".join(a[max(0, i1 - 4):i1]) + " [...] " + " ".join(a[i2:i2 + 4])
        out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2]), context))
    return out


def highlighted(raw: str, refined: str) -> str:
    """Markdown text of the refined transcript with ~~old~~ **new** at each change."""
    a, b = raw.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out += a[i1:i2]
        else:
            old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            out.append((f"~~{old}~~ " if old else "") + (f"**{new}**" if new else ""))
    return " ".join(out)
