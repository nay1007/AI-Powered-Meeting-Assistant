"""Export: one MeetingRecord -> ten files (two transcripts, the full record, and
minutes / key decisions / action items, each as Markdown and JSON).

Every file is built from record.to_dict(), the same data, so Markdown and JSON agree.
"""
import json
import os

from .schema import MeetingRecord

FILES = (
    "raw_transcript.txt", "refined_transcript.txt",
    "meeting_record.md", "meeting_record.json",
    "minutes.md", "minutes.json",
    "key_decisions.md", "key_decisions.json",
    "action_items.md", "action_items.json",
)


def to_json(record: MeetingRecord) -> str:
    return json.dumps(record.to_dict(), indent=2, ensure_ascii=False)


# Each section is built ONCE here and reused by the full record and by its own file,
# so a section's content is identical wherever it appears.
def _summary_lines(d: dict) -> list[str]:
    return ["## Summary", "", d["summary"], ""]


def _minutes_lines(d: dict) -> list[str]:
    out = ["## Minutes", ""]
    if d["minutes"]:
        for topic in d["minutes"]:
            out += [f"### {topic['topic']}", ""] + [f"- {p}" for p in topic["points"]] + [""]
    else:
        out += ["None.", ""]
    return out


def _decisions_lines(d: dict) -> list[str]:
    items = [f"{i}. {x['decision']}  \n   > {x['evidence']}" for i, x in enumerate(d["decisions"], 1)]
    return ["## Key Decisions", ""] + (items or ["None."])


def _actions_lines(d: dict) -> list[str]:
    out = ["## Action Items", ""]
    if not d["action_items"]:
        return out + ["None."]
    out += ["| # | Task | Owner | Deadline | Evidence |", "|---|---|---|---|---|"]
    for i, a in enumerate(d["action_items"], 1):
        cells = [a["task"], a["owner"], a["deadline"], a["evidence"]]
        out.append(f"| {i} | " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return out


def to_markdown(record: MeetingRecord) -> str:
    d = record.to_dict()
    out = ["# Meeting Record", ""]

    meta = d["metadata"]
    if meta:
        out += ["## Details", ""] + [f"- **{k}**: {v}" for k, v in meta.items()] + [""]

    out += _summary_lines(d) + _minutes_lines(d) + _decisions_lines(d)

    out += ["", "## Proposals (not agreed)", ""]
    out += [f"{i}. {x['proposal']}  \n   > {x['evidence']}" for i, x in enumerate(d["proposals"], 1)] or ["None."]

    out += [""] + _actions_lines(d)

    out += ["", "## Warnings", ""] + ([f"- {w}" for w in d["warnings"]] or ["None."])
    out += ["", "## Refined Transcript", "", d["refined_transcript"], "",
            "## Raw Transcript", "", d["raw_transcript"], ""]
    return "\n".join(out)


def _section_markdown(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", ""] + lines) + "\n"


def _section_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def save_outputs(record: MeetingRecord, out_dir: str) -> dict[str, str]:
    """Write every file in FILES; return {file_name: full_path}."""
    os.makedirs(out_dir, exist_ok=True)
    d = record.to_dict()
    contents = {
        "raw_transcript.txt": record.raw_transcript + "\n",
        "refined_transcript.txt": record.refined_transcript + "\n",
        "meeting_record.md": to_markdown(record),
        "meeting_record.json": to_json(record) + "\n",
        "minutes.md": _section_markdown("Meeting Minutes", _summary_lines(d) + _minutes_lines(d)),
        "minutes.json": _section_json({"summary": d["summary"], "minutes": d["minutes"]}),
        "key_decisions.md": _section_markdown("Key Decisions", _decisions_lines(d)),
        "key_decisions.json": _section_json({"decisions": d["decisions"]}),
        "action_items.md": _section_markdown("Action Items", _actions_lines(d)),
        "action_items.json": _section_json({"action_items": d["action_items"]}),
    }
    paths = {}
    for name, text in contents.items():
        paths[name] = os.path.join(out_dir, name)
        with open(paths[name], "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return paths
