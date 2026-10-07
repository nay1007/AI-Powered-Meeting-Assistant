"""Thin wrappers around ffmpeg: find it, decode audio, measure loudness."""
import re
import shutil
import subprocess

from .errors import PipelineError


def ffmpeg_path() -> str:
    """Use a system ffmpeg if present, else the one bundled by imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise PipelineError(
            "ffmpeg was not found. Install it (winget install Gyan.FFmpeg) "
            "or run: pip install imageio-ffmpeg"
        ) from exc


def probe(path: str) -> tuple[float, float]:
    """Decode the whole file. Return (duration_seconds, max_volume_dB).

    Raises PipelineError if ffmpeg cannot decode the file.
    """
    cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-i", path,
           "-vn", "-af", "volumedetect", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Reading the audio file took too long; it may be corrupt.") from exc
    log = proc.stderr
    if proc.returncode != 0 or "Audio:" not in log:
        raise PipelineError("The file could not be read as audio. It may be corrupt or not a real audio file.")

    dur = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", log)
    if not dur:
        raise PipelineError("The file contains no decodable audio.")
    h, m, s = dur[-1]
    seconds = int(h) * 3600 + int(m) * 60 + float(s)

    vol = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?|-inf)\s*dB", log)
    max_db = float("-inf") if (vol is None or vol.group(1) == "-inf") else float(vol.group(1))
    return seconds, max_db
