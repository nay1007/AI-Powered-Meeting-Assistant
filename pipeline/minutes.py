"""Stage 3: refined transcript -> summary, minutes, decisions, proposals, action items.

LLM #2 (Qwen3 on Groq by default, see config.MINUTES_MODEL) is a different model
with its own prompt (prompts/minutes_prompt.txt) and never sees the raw transcript
or the stage-2 prompt. Its JSON answer is checked in plain code:
  * invalid or wrongly shaped JSON -> the model is told what is wrong and retried
    (up to MAX_RETRIES times); if it still fails we stop with a clear error
    instead of making something up.
  * an owner or deadline whose words never appear in the transcript is reset
    to "Unspecified".
  * quotes and hedged "decisions" that look suspicious produce warnings.

Very long transcripts are split into parts of about CHUNK_WORDS words (whole
sentences), each part gets its own record, and the records are merged.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field

from .config import MINUTES_MODEL, require_key
from .errors import PipelineError
from .refine import split_into_chunks

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "minutes_prompt.txt")
MAX_RETRIES = 2          # invalid JSON: first try + 2 corrected retries
CHUNK_WORDS = 2500       # transcripts longer than this are processed in parts
UNSPECIFIED = "Unspecified"

# words ignored when checking that an owner / deadline appears in the transcript
FILLER = {"by", "on", "the", "of", "before", "at", "in", "to", "end", "next", "this", "a", "an",
          "and", "until", "till", "for", "from", "st", "nd", "rd", "th"}
# "owners" that are not a person or a named team
VAGUE_OWNERS = {"i", "me", "myself", "we", "us", "you", "speaker", "someone", "somebody", "anyone",
                "everyone", "unknown", "n/a", "na", "none", "tbd", "tba", "unassigned", "team"}
HEDGES = ("maybe", "perhaps", "might", "could we", "what if", "i suggest", "we should consider",
          "not agreed", "haven't agreed", "have not agreed", "nobody has agreed", "no one has agreed")

SUMMARY_PROMPT = (
    "You combine partial summaries of consecutive parts of ONE meeting into a single summary of "
    "3-5 sentences giving the big picture and overall outcome. Use only facts present in the "
    "partial summaries. Do not add names, numbers or decisions. Output only the summary text.")


@dataclass
class MinutesResult:
    record: dict
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1
    parts: int = 1


def load_prompt() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------------------------------------------- validation
def _is_str(x) -> bool:
    return isinstance(x, str)


def validate_record(data) -> list[str]:
    """Return a list of structure problems (empty list = valid)."""
    if not isinstance(data, dict):
        return ["The top level must be a JSON object."]
    errors = []
    if not _is_str(data.get("summary")) or not data["summary"].strip():
        errors.append("'summary' must be a non-empty string.")

    def need_list(key, fields):
        items = data.get(key)
        if not isinstance(items, list):
            errors.append(f"'{key}' must be a list (use [] if there are none).")
            return
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"'{key}[{i}]' must be an object.")
                continue
            for f in fields:
                if not _is_str(item.get(f)):
                    errors.append(f"'{key}[{i}].{f}' must be a string.")

    need_list("decisions", ["decision", "evidence"])
    need_list("proposals", ["proposal", "evidence"])
    need_list("action_items", ["task", "owner", "deadline", "evidence"])

    minutes = data.get("minutes")
    if not isinstance(minutes, list):
        errors.append("'minutes' must be a list of {topic, points}.")
    else:
        for i, t in enumerate(minutes):
            if not isinstance(t, dict) or not _is_str(t.get("topic")):
                errors.append(f"'minutes[{i}].topic' must be a string.")
            elif not isinstance(t.get("points"), list) or not all(_is_str(p) for p in t["points"]):
                errors.append(f"'minutes[{i}].points' must be a list of strings.")
    return errors


def _parse_json(text: str):
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()   # reasoning models
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    return json.loads(t)


# ------------------------------------------------------------ truth checks
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def appears_in(value: str, transcript_words: set[str]) -> bool:
    """True if every meaningful word of value occurs in the transcript."""
    needed = [w for w in _words(value) if w not in FILLER]
    return bool(needed) and all(w in transcript_words for w in needed)


def sanitize_owners_and_deadlines(record: dict, transcript: str) -> list[str]:
    """Reset owner/deadline to 'Unspecified' unless the transcript really says it."""
    seen = set(_words(transcript))
    notes = []
    for i, item in enumerate(record["action_items"], 1):
        for key in ("owner", "deadline"):
            value = item[key].strip()
            if not value or value.lower() == UNSPECIFIED.lower():
                item[key] = UNSPECIFIED
                continue
            bad = not appears_in(value, seen)
            if key == "owner" and value.lower().strip(". ") in VAGUE_OWNERS:
                bad = True
            if bad:
                notes.append(f"Action item {i}: {key} '{value}' is not stated in the transcript; reset to {UNSPECIFIED}.")
                item[key] = UNSPECIFIED
    return notes


def _squash(text: str) -> str:
    return " ".join(_words(text))


def check_evidence(record: dict, transcript: str) -> list[str]:
    """Warn when a quote is not really in the transcript, or a 'decision' sounds hedged."""
    body = _squash(transcript)
    warnings = []
    for key, label in (("decisions", "Decision"), ("proposals", "Proposal"), ("action_items", "Action item")):
        for i, item in enumerate(record[key], 1):
            quote = _squash(item["evidence"])
            if not quote or quote not in body:
                warnings.append(f"{label} {i}: its supporting quote was not found word-for-word in the transcript.")
    for i, item in enumerate(record["decisions"], 1):
        text = (item["decision"] + " " + item["evidence"]).lower()
        if any(h in text for h in HEDGES):
            warnings.append(f"Decision {i} contains hedging words ('maybe', 'not agreed'...); it may be a proposal.")
    return warnings


# ---------------------------------------------------------------- the stage
def _call_llm(client, messages: list[dict], json_mode: bool = True) -> str:
    last = None
    for attempt in range(3):
        try:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = client.chat.completions.create(
                model=MINUTES_MODEL, messages=messages, temperature=0.0, **kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # network, rate limit...
            last = exc
            time.sleep(5 * (attempt + 1))
    raise PipelineError(f"Minutes generation failed after 3 tries: {last}")


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _generate_part(client, transcript: str) -> tuple[dict, list[str], int]:
    """One LLM-checked record for one piece of transcript."""
    messages = [_msg("system", load_prompt()), _msg("user", f"<transcript>\n{transcript}\n</transcript>")]
    problems: list[str] = []
    for attempt in range(1, MAX_RETRIES + 2):
        answer = _call_llm(client, messages)
        try:
            data = _parse_json(answer)
            problems = validate_record(data)
        except json.JSONDecodeError as exc:
            data, problems = None, [f"Your output was not valid JSON ({exc})."]
        if not problems:
            break
        messages += [_msg("assistant", answer), _msg(
            "user", "Your output was rejected:\n- " + "\n- ".join(problems) +
                    "\nReturn the corrected record as valid JSON in exactly the required shape, nothing else.")]
    else:
        raise PipelineError(
            "The meeting record could not be generated: the model kept returning an invalid answer. "
            "Please try again. Last problem: " + problems[0])

    record = {k: data[k] for k in ("summary", "minutes", "decisions", "proposals", "action_items")}
    warnings = sanitize_owners_and_deadlines(record, transcript)
    warnings += check_evidence(record, transcript)
    return record, warnings, attempt


def _dedupe(items: list[dict], key: str) -> list[dict]:
    seen, out = set(), []
    for item in items:
        k = _squash(item[key])
        if k and k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def merge_records(records: list[dict], summary: str) -> dict:
    """Combine the records of consecutive transcript parts into one."""
    return {
        "summary": summary,
        "minutes": [t for r in records for t in r["minutes"]],
        "decisions": _dedupe([d for r in records for d in r["decisions"]], "decision"),
        "proposals": _dedupe([p for r in records for p in r["proposals"]], "proposal"),
        "action_items": _dedupe([a for r in records for a in r["action_items"]], "task"),
    }


def _combine_summaries(client, summaries: list[str]) -> str:
    joined = " ".join(summaries)
    try:
        text = _call_llm(client, [_msg("system", SUMMARY_PROMPT), _msg(
            "user", "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(summaries, 1)))], json_mode=False)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text or joined
    except PipelineError:
        return joined    # never lose the content just because the final polish failed


def generate_minutes(refined_transcript: str, client=None) -> MinutesResult:
    if not refined_transcript.strip():
        raise PipelineError("There is no transcript to summarise.")
    if client is None:
        from groq import Groq
        client = Groq(api_key=require_key("GROQ_API_KEY"))

    pieces = split_into_chunks(refined_transcript, CHUNK_WORDS)
    records, warnings, attempts = [], [], 0
    for n, piece in enumerate(pieces, 1):
        record, part_warnings, used = _generate_part(client, piece)
        records.append(record)
        attempts = max(attempts, used)
        prefix = f"Part {n}: " if len(pieces) > 1 else ""
        warnings += [prefix + w for w in part_warnings]

    if len(records) == 1:
        return MinutesResult(records[0], warnings, attempts, 1)
    summary = _combine_summaries(client, [r["summary"] for r in records])
    return MinutesResult(merge_records(records, summary), warnings, attempts, len(records))
