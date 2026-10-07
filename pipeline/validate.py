"""Stage 0: check the uploaded file before spending any model calls on it."""
import os
from dataclasses import dataclass

from .audio_tools import probe
from .errors import PipelineError

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac", ".mpeg"}
MIN_SECONDS = 1.0
SILENCE_DB = -50.0  # loudest point quieter than this => treated as silence


@dataclass
class AudioInfo:
    path: str
    duration_seconds: float
    max_volume_db: float


def validate_audio(path: str) -> AudioInfo:
    """Return AudioInfo, or raise PipelineError with a clear message."""
    if not path or not os.path.isfile(path):
        raise PipelineError("No file was provided.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise PipelineError(f"Unsupported file type '{ext or 'none'}'. Please upload one of: {allowed}.")

    if os.path.getsize(path) == 0:
        raise PipelineError("The file is empty (0 bytes).")

    seconds, max_db = probe(path)

    if seconds < MIN_SECONDS:
        raise PipelineError("The recording is too short to contain a meeting.")
    if max_db <= SILENCE_DB:
        raise PipelineError("The recording is silent: no speech could be detected.")

    return AudioInfo(path=path, duration_seconds=seconds, max_volume_db=max_db)
