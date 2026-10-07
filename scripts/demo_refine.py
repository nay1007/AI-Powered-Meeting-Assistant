"""Developer demo: audio -> raw transcript -> refined transcript, with a change list.

Usage:  python -m scripts.demo_refine samples/test_meeting.wav
"""
import sys

from pipeline.refine import highlighted, refine, word_changes
from pipeline.transcribe import transcribe
from pipeline.validate import validate_audio

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]
info = validate_audio(path)
print(f"File OK: {info.duration_seconds:.1f} s of audio\n")

raw = transcribe(path)
print("=== RAW TRANSCRIPT (Whisper) ===\n" + raw + "\n")

result = refine(raw)
print("=== REFINED TRANSCRIPT (LLM #1) ===\n" + result.refined + "\n")

print("=== SAFETY CHECK REPORT ===")
for c in result.chunks:
    print(f"chunk {c.index}: {c.status} after {c.attempts} attempt(s)", *c.problems, sep="\n   ")

changes = word_changes(raw, result.refined)
print(f"\n=== EVERY CHANGE ({len(changes)}) ===")
for i, (old, new, ctx) in enumerate(changes, 1):
    print(f"{i}. '{old}'  ->  '{new}'      (context: {ctx})")

print("\n=== BEFORE/AFTER, CHANGES HIGHLIGHTED (~~old~~ **new**) ===\n" + highlighted(raw, result.refined))
