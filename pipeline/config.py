"""Central settings: model names and API keys. Keys come from the .env file."""
import os

from dotenv import load_dotenv

from .errors import PipelineError

load_dotenv()

# Stage 1: speech-to-text
STT_MODEL = "whisper-large-v3"
# Stage 2: transcript refinement (LLM #1)
REFINE_MODEL = "openai/gpt-oss-120b"
# Stage 3: minutes, decisions, action items (LLM #2, a different model and company)
MINUTES_MODEL = "gemini-2.5-flash"


def require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PipelineError(
            f"Missing {name}. Copy .env.example to .env and paste your key after '{name}='."
        )
    return value
