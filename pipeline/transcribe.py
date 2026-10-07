"""Stage 1: audio -> raw transcript, using Whisper large-v3 on Groq.

Audio is converted losslessly (FLAC, mono, 16 kHz: exactly what Whisper uses)
and cut into ~10-minute pieces so each upload stays under the free API limit.

Cutting rule:
  1. Aim for a cut every ~10 minutes.
  2. Look +/-30 s around the target and cut in the middle of the LONGEST pause.
  3. If there is no pause there, cut at the target but let the two pieces
     overlap by a few seconds, so no word is lost. The repeated words are
     removed when the transcripts are joined.
Transcripts are then joined in order.
"""
import os
import re
import subprocess
import tempfile
import time

from .audio_tools import ffmpeg_path, probe
from .config import STT_MODEL, require_key
from .errors import PipelineError

TARGET_CHUNK_SECONDS = 600
SEARCH_WINDOW_SECONDS = 30
OVERLAP_SECONDS = 4
MIN_LEFTOVER_SECONDS = 60       # a shorter tail is merged into the last piece
MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # free limit is 25 MB
MAX_OVERLAP_WORDS = 60
MIN_OVERLAP_WORDS = 3


def _run(cmd: list[str], error: str) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(error)
    return proc.stderr


def to_flac(path: str, out_path: str, start: float | None = None, end: float | None = None) -> None:
    """Lossless mono 16 kHz FLAC, optionally only the part from start to end."""
    cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", path]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", out_path]
    _run(cmd, "The audio could not be converted for transcription.")


def find_silences(path: str) -> list[tuple[float, float]]:
    """(start, end) of every pause of 0.3 s or more."""
    log = _run([ffmpeg_path(), "-hide_banner", "-nostdin", "-i", path,
                "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
               "The audio could not be analysed for pauses.")
    starts = [float(x) for x in re.findall(r"silence_start: (-?\d+(?:\.\d+)?)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?\d+(?:\.\d+)?)", log)]
    return [(max(s, 0.0), e) for s, e in zip(starts, ends)]


def choose_cuts(duration: float, silences: list[tuple[float, float]]) -> list[tuple[float, bool]]:
    """Return [(cut_time, overlapped)] following the cutting rule above."""
    cuts: list[tuple[float, bool]] = []
    target = float(TARGET_CHUNK_SECONDS)
    while target < duration - MIN_LEFTOVER_SECONDS:
        lo, hi = target - SEARCH_WINDOW_SECONDS, target + SEARCH_WINDOW_SECONDS
        inside = [(s, e) for s, e in silences if lo <= (s + e) / 2 <= hi]
        if inside:
            s, e = max(inside, key=lambda p: p[1] - p[0])  # longest pause
            cut, overlapped = (s + e) / 2, False
        else:
            cut, overlapped = target, True
        cuts.append((cut, overlapped))
        target = cut + TARGET_CHUNK_SECONDS
    return cuts


def split_audio(path: str, out_dir: str) -> tuple[list[str], list[bool]]:
    """Return (piece_files, overlapped_flags).

    overlapped_flags[i] is True if piece i+1 overlaps the end of piece i.
    """
    full = os.path.join(out_dir, "full.flac")
    to_flac(path, full)
    duration, _ = probe(full)
    cuts = choose_cuts(duration, find_silences(full)) if duration > TARGET_CHUNK_SECONDS + MIN_LEFTOVER_SECONDS else []

    starts = [0.0] + [c for c, _ in cuts]
    ends = [c + (OVERLAP_SECONDS if ov else 0.0) for c, ov in cuts] + [duration]
    pieces = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        piece = os.path.join(out_dir, f"chunk_{i:03d}.flac")
        if len(cuts) == 0:
            piece = full
        else:
            to_flac(full, piece, start=a, end=b)
        if os.path.getsize(piece) > MAX_UPLOAD_BYTES:
            raise PipelineError("A piece of the audio is too large to upload. Try a shorter recording.")
        pieces.append(piece)
    return pieces, [ov for _, ov in cuts]


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def merge_transcripts(parts: list[str], overlapped: list[bool]) -> str:
    """Join transcripts in order; drop words repeated across overlapping cuts."""
    merged = parts[0].split() if parts else []
    for i, part in enumerate(parts[1:]):
        words = part.split()
        if overlapped[i]:
            tail = [_norm(w) for w in merged[-MAX_OVERLAP_WORDS:]]
            for n in range(min(MAX_OVERLAP_WORDS, len(tail), len(words)), MIN_OVERLAP_WORDS - 1, -1):
                if tail[-n:] == [_norm(w) for w in words[:n]]:
                    words = words[n:]
                    break
        merged += words
    return " ".join(merged).strip()


def _transcribe_chunk(client, chunk_path: str) -> str:
    last_error = None
    for attempt in range(3):
        try:
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=(os.path.basename(chunk_path), f.read()),
                    model=STT_MODEL,
                    language="en",
                    temperature=0.0,
                    response_format="text",
                )
            return (result if isinstance(result, str) else result.text).strip()
        except Exception as exc:  # network, rate limit, bad key...
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise PipelineError(f"Speech-to-text failed after 3 tries: {last_error}")


def transcribe(path: str, client=None) -> str:
    """Return the raw transcript for a (already validated) audio file."""
    if client is None:
        from groq import Groq
        client = Groq(api_key=require_key("GROQ_API_KEY"))

    with tempfile.TemporaryDirectory() as tmp:
        files, overlapped = split_audio(path, tmp)
        parts = [_transcribe_chunk(client, f) for f in files]

    text = merge_transcripts(parts, overlapped)
    if not text:
        raise PipelineError("No speech could be recognised in the recording.")
    return text
