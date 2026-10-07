"""Developer entry point: audio file -> the four output files.

Usage:  python -m scripts.run_pipeline samples/test_meeting.wav [output_folder]
"""
import sys

from pipeline.errors import PipelineError
from pipeline.run import run_pipeline

sys.stdout.reconfigure(encoding="utf-8")
try:
    result = run_pipeline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, on_status=print)
except PipelineError as exc:
    sys.exit(f"ERROR: {exc}")

print("\nFiles produced:")
for name, path in result.files.items():
    print(f"  {name}: {path}")
print("\nWarnings:", *(result.record.warnings or ["none"]), sep="\n  ")
