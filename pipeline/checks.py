"""Safety checks that compare the text before and after LLM refinement.

These are plain code (no AI), so they cannot be talked out of their job.
"""
import difflib
import re
from collections import Counter

MAX_SPAN_WORDS = 6         # one change may replace at most this many words (a term fix is short)
MAX_CHANGED_FRACTION = 0.20  # at most 20 % of the original words may be touched in total
MIN_WORDS_FOR_FRACTION = 50  # ... but a percentage is meaningless on tiny texts
SENTENCE_TOLERANCE = 0.05  # sentence count may change by at most 5 % (min +/-1)

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "half", "quarter", "double", "triple", "twice",
}
NEGATION_WORDS = {"not", "no", "never", "cannot", "none", "nothing", "nobody", "neither", "nor", "without"}

_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def numbers_in(text: str) -> Counter:
    """Every number, written as digits (commas ignored) or as a number word."""
    found = Counter(m.group().replace(",", "").rstrip(",.") for m in _DIGITS.finditer(text))
    for w in _WORD.findall(text.lower()):
        if w in NUMBER_WORDS:
            found[w] += 1
    return found


def negations_in(text: str) -> Counter:
    """Count of not/no/never/... and every contraction ending in n't."""
    found = Counter()
    for w in _WORD.findall(text.lower()):
        if w in NEGATION_WORDS:
            found[w] += 1
        elif w.endswith("n't"):
            found["n't"] += 1
    return found


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))


def _diff(before: Counter, after: Counter) -> str:
    parts = []
    for key in sorted(set(before) | set(after)):
        if before[key] != after[key]:
            parts.append(f"'{key}' appears {before[key]}x in the original but {after[key]}x in your output")
    return "; ".join(parts)


def _change_size_problems(original: str, refined: str) -> list[str]:
    """Fixing a term touches a few words. Rewriting or deleting text touches many."""
    a, b = original.split(), refined.split()
    problems, changed = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        changed += i2 - i1
        if max(i2 - i1, j2 - j1) > MAX_SPAN_WORDS:
            problems.append(
                f"One change is too large: '{' '.join(a[i1:i2])[:80]}' became '{' '.join(b[j1:j2])[:80]}'. "
                f"A fix may replace at most {MAX_SPAN_WORDS} words. Do not summarize, rewrite, add or remove sentences."
            )
    if len(a) >= MIN_WORDS_FOR_FRACTION and changed / len(a) > MAX_CHANGED_FRACTION:
        problems.append(
            f"Too much of the text changed ({changed} of {len(a)} words). Only misheard technical terms may change."
        )
    return problems


def find_problems(original: str, refined: str) -> list[str]:
    """Return a list of plain-English problems; empty list means the output is safe."""
    problems = []

    d = _diff(numbers_in(original), numbers_in(refined))
    if d:
        problems.append(f"Numbers changed: {d}. Every number must stay exactly as in the original.")

    d = _diff(negations_in(original), negations_in(refined))
    if d:
        problems.append(f"Negations changed: {d}. Negations must never be added or removed.")

    problems += _change_size_problems(original, refined)

    os_, rs = sentence_count(original), sentence_count(refined)
    if abs(rs - os_) > max(1, SENTENCE_TOLERANCE * os_):
        problems.append(
            f"Sentence count changed: the original has {os_} sentences but your output has {rs}. "
            "Do not merge, split, add or remove sentences."
        )
    return problems
