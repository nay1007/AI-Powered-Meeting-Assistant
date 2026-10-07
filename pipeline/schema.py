"""The ONE data object that holds everything the pipeline produces.

Markdown and JSON are both generated from MeetingRecord.to_dict(), so they can
never disagree.
"""
from dataclasses import dataclass, field


@dataclass
class MeetingRecord:
    raw_transcript: str
    refined_transcript: str
    summary: str
    minutes: list[dict]          # [{topic, points}]
    decisions: list[dict]        # [{decision, evidence}]
    proposals: list[dict]        # [{proposal, evidence}]
    action_items: list[dict]     # [{task, owner, deadline, evidence}]
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # models used, durations, refinement report

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "raw_transcript": self.raw_transcript,
            "refined_transcript": self.refined_transcript,
            "summary": self.summary,
            "minutes": self.minutes,
            "decisions": self.decisions,
            "proposals": self.proposals,
            "action_items": self.action_items,
            "warnings": self.warnings,
        }
