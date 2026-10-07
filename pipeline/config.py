"""Central settings: model names and API keys. Keys come from the .env file."""
import os

from dotenv import load_dotenv

from .errors import PipelineError

load_dotenv()

# Stage 1: speech-to-text. Runs LOCALLY with faster-whisper (open-source Whisper),
# so there is no API, no key and no usage limit. Override any of these in .env.
STT_MODEL = os.getenv("WHISPER_MODEL", "").strip() or "large-v3"        # tiny/base/small/medium/large-v3
STT_DEVICE = os.getenv("WHISPER_DEVICE", "").strip() or "cpu"           # "cpu" or "cuda" (NVIDIA GPU)
STT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "").strip() or "int8"  # "int8" for CPU, "float16" for GPU
# CPU threads for Whisper. The engine defaults to only 4, which wastes most of a modern CPU.
# Default: this machine's core count, capped at 16. Override with WHISPER_CPU_THREADS in .env.
STT_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "").strip() or min(16, os.cpu_count() or 4))
# Stage 2: transcript refinement (LLM #1)
REFINE_MODEL = "openai/gpt-oss-120b"
# Stage 3: minutes, decisions, action items (LLM #2, a different model and company
# from stage 2). Hosted on Groq. Override with MINUTES_MODEL in .env, e.g. a Llama 4
# id such as "meta-llama/llama-4-scout-17b-16e-instruct" if your Groq account has it.
MINUTES_MODEL = os.getenv("MINUTES_MODEL", "").strip() or "qwen/qwen3.8-27b"


def require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PipelineError(
            f"Missing {name}. Copy .env.example to .env and paste your key after '{name}='."
        )
    return value
