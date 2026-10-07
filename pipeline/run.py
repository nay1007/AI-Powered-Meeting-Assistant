"""The whole pipeline in one function: audio path in, all outputs out.

Order: validate -> transcribe -> refine -> document (minutes), then export.
"""
import os
from dataclasses import dataclass

from .config import MINUTES_MODEL, REFINE_MODEL, STT_MODEL
from .errors import PipelineError
from .export import save_outputs
from .minutes import generate_minutes
from .refine import refine, word_changes
from .schema import MeetingRecord
from .transcribe import transcribe
from .validate import validate_audio


@dataclass
class PipelineResult:
    record: MeetingRecord
    files: dict[str, str]


def run_pipeline(audio_path: str, out_dir: str | None = None, on_status=None,
                 groq_client=None) -> PipelineResult:
    """Run every stage on one recording and save the four output files.

    on_status(message) is called as each stage starts (for a UI progress display).
    Raises PipelineError with a user-readable message on any failure.
    """
    say = on_status or (lambda msg: None)
    try:
        say("Checking the audio file...")
        info = validate_audio(audio_path)

        say("Stage 1/3: transcribing speech...")
        raw = transcribe(audio_path)

        say("Stage 2/3: refining the transcript...")
        refined = refine(raw, client=groq_client)

        say("Stage 3/3: writing minutes, decisions and action items...")
        result = generate_minutes(refined.refined, client=groq_client)
    except PipelineError:
        raise
    except Exception as exc:   # never show a raw crash to the user
        raise PipelineError(f"Something unexpected went wrong: {exc}") from exc

    warnings = list(result.warnings)
    for c in refined.chunks:
        if c.status == "fell_back_to_raw":
            warnings.append(f"Refinement part {c.index + 1} was rejected by the safety checks "
                            f"and left as the raw transcript ({'; '.join(c.problems)}).")

    record = MeetingRecord(
        raw_transcript=raw,
        refined_transcript=refined.refined,
        warnings=warnings,
        metadata={
            "source_file": os.path.basename(audio_path),
            "duration_seconds": round(info.duration_seconds, 1),
            "speech_to_text_model": f"faster-whisper {STT_MODEL} (local)",
            "refinement_model": REFINE_MODEL,
            "minutes_model": MINUTES_MODEL,
            "refinement_corrections": len(word_changes(raw, refined.refined)),
            "refinement_parts_fell_back_to_raw": sum(c.status == "fell_back_to_raw" for c in refined.chunks),
            "minutes_parts": result.parts,
        },
        **result.record,
    )
    if out_dir is None:
        out_dir = os.path.join("outputs", os.path.splitext(os.path.basename(audio_path))[0])
    say("Saving output files...")
    return PipelineResult(record, save_outputs(record, out_dir))
