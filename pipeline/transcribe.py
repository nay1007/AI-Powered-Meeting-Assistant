"""Stage 1: audio -> raw transcript, using open-source Whisper that runs on THIS computer.

We use faster-whisper (the same Whisper models as OpenAI, run through a faster engine).
There is no API, no key and no upload limit. The model files are downloaded once
(about 3 GB for large-v3) the first time this stage runs and are then reused.

The audio is first converted to lossless mono 16 kHz FLAC (exactly what Whisper
uses). faster-whisper then handles recordings of any length itself: it skips
silences and works through the audio in 30-second windows, so we do not need to
cut the file ourselves.
"""
import subprocess
import tempfile
import os

from .audio_tools import ffmpeg_path
from .config import STT_COMPUTE_TYPE, STT_CPU_THREADS, STT_DEVICE, STT_MODEL
from .errors import PipelineError

# Whisper copies the writing style of the text it starts from. Without any, the first
# 30 seconds often come out lowercase with no punctuation. This neutral, punctuated
# sentence fixes that. It contains no names, numbers or jargon, so it cannot leak
# wrong facts into the transcript.
PUNCTUATION_PRIMER = "Hello, everyone. Thanks for joining. Let's get started, shall we?"

_model = None   # loaded once and reused: loading a large model takes a while


def to_flac(path: str, out_path: str) -> None:
    """Lossless mono 16 kHz FLAC copy of the recording."""
    cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", path,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", out_path]
    if subprocess.run(cmd, capture_output=True, text=True).returncode != 0:
        raise PipelineError("The audio could not be converted for transcription.")


def load_model():
    """Load (and on first use download) the Whisper model."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE,
                                  cpu_threads=STT_CPU_THREADS)
        except Exception as exc:
            raise PipelineError(
                f"The speech-to-text model '{STT_MODEL}' could not be loaded: {exc}. "
                "The first run needs an internet connection to download it.") from exc
    return _model


def transcribe(path: str, model=None) -> str:
    """Return the raw transcript for a (already validated) audio file."""
    model = model or load_model()
    with tempfile.TemporaryDirectory() as tmp:
        flac = os.path.join(tmp, "audio.flac")
        to_flac(path, flac)
        try:
            segments, _info = model.transcribe(
                flac,
                language="en",
                beam_size=5,
                temperature=0.0,
                initial_prompt=PUNCTUATION_PRIMER,
                vad_filter=True,                    # skip silence; avoids invented text in pauses
                condition_on_previous_text=False,   # stops one mistake repeating through the rest
            )
            text = " ".join(s.text.strip() for s in segments)   # segments are produced lazily here
        except Exception as exc:
            raise PipelineError(f"Speech-to-text failed: {exc}") from exc

    text = " ".join(text.split())
    if not text:
        raise PipelineError("No speech could be recognised in the recording.")
    return text
