# AI-Powered Meeting Assistant

Upload a recording of an English meeting and get back a raw transcript, a refined transcript, meeting minutes, key decisions and action items. Everything is produced by running models on your audio in one pipeline behind a simple web interface. Nothing is pre-written.


## Models and their roles

| Stage | Role | Model | Where it runs |
|---|---|---|---|
| 1 | Speech-to-text (raw transcript) | `large-v3` through [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (open-source Whisper) | Locally on your computer, no API or key |
| 2 | **LLM #1**: fix misheard technical terms (refined transcript) | `openai/gpt-oss-120b` | Groq API (free tier) |
| 3 | **LLM #2**: summary, minutes, decisions, action items | `qwen/qwen3.8-27b` | Groq API (free tier) |

###### NOTE : large-v3 is around a 3 GB model and required a dedicated GPU for fast computation of transcripts. If running on CPU, large files will take long time.

The two language models are different models from different makers, called in separate stages with separate prompts. Stage 3 never sees the raw transcript or the stage 2 prompt.

## How it works

```
audio file
   |
   v
[0] Validate ---- empty / wrong type / corrupt / too short / silent -> clear error message
   |
   v
[1] Speech-to-text (faster-whisper, local) ------------------------> raw transcript
   |
   v
[2] Refinement (LLM #1, prompts/refine_prompt.txt)
       -> safety checks in plain code; retry with the reason; fall back to raw text
   |  -----------------------------------------------------------> refined transcript
   v
[3] Documentation (LLM #2, prompts/minutes_prompt.txt)
       -> JSON validation; owner/deadline check; quote check
   |  -----------------------------------------------------------> summary, minutes,
   v                                                                decisions, proposals,
[Export] Markdown + JSON from one shared record                     action items
```

### Stage 0: validation (`pipeline/validate.py`)
Rejects missing, wrong-type, empty, undecodable, shorter than 1 second, and silent files with a plain-language message instead of a crash. Accepted types: `.mp3 .wav .m4a .mp4 .flac .ogg .webm .aac .mpeg`.

### Stage 1: transcription (`pipeline/transcribe.py`)
The audio is converted to lossless mono 16 kHz FLAC (what Whisper uses internally) and passed to faster-whisper with English forced, temperature 0, and a voice-activity filter that skips silence. A short punctuated priming sentence makes the first seconds come out capitalised and punctuated. The model is loaded once and reused.

### Stage 2: refinement (`pipeline/refine.py`, `pipeline/checks.py`)
The model's only job is to fix plausible speech-recognition errors in technical terms, acronyms and jargon (for example "cooper net ease" to "Kubernetes"). The prompt forbids changing names, numbers, negations and commitments, forbids summarising or reordering, and says to leave text unchanged when unsure.

Because an LLM can break those rules without warning, plain code compares the input and output of every chunk (about 800 words, cut at sentence boundaries):

- the same numbers (digits and number words) must appear the same number of times
- the same negations (`not, no, never, n't, cannot, none, nothing, ...`) must appear the same number of times
- no single change may replace more than 6 words, and no more than 20% of a chunk (50+ words) may change
- the sentence count may change by at most 5% (minimum 1)

If a check fails, the model is told exactly what went wrong and asked again (3 attempts in total). If it still fails, that chunk keeps its raw text and a warning is shown. A bad refinement therefore cannot reach the final record. In the web app, each correction is highlighted inside the refined transcript, and hovering shows the original wording.

### Stage 3: minutes, decisions and action items (`pipeline/minutes.py`)
The refined transcript goes to a second, separate LLM that returns structured JSON:

| Field | Meaning |
|---|---|
| `summary` | 3 to 5 sentence overview |
| `minutes` | topics, each with discussion points |
| `decisions` | only things clearly agreed or confirmed |
| `proposals` | suggestions nobody agreed to; never mixed into decisions |
| `action_items` | concrete work someone committed to or was explicitly assigned: `task`, `owner`, `deadline` |

Every decision, proposal and action item carries a short verbatim `evidence` quote. The prompt contains the definitions of decision, proposal and action item, plus a short worked example. Code then enforces:

- invalid or wrongly shaped JSON is sent back with the problem listed, up to 2 retries, then a clear error
- an owner or deadline whose words never appear in the transcript, or an owner like "I" or "someone", is reset to exactly `Unspecified`
- warnings for quotes that are not word-for-word in the transcript, and for "decisions" containing hedging words such as "maybe" or "not agreed"
- transcripts over about 2500 words are processed in parts, merged and de-duplicated, with a combined summary

### Outputs
Every run writes these files to `outputs/<recording name>/`. Markdown and JSON are built from the same record, so they carry the same content.

| Output | Files |
|---|---|
| Raw transcript | `raw_transcript.txt` |
| Refined transcript | `refined_transcript.txt` |
| Meeting minutes | `minutes.md`, `minutes.json` |
| Key decisions | `key_decisions.md`, `key_decisions.json` |
| Action items | `action_items.md`, `action_items.json` |
| Whole record (also proposals, warnings, model names) | `meeting_record.md`, `meeting_record.json` |

Empty lists are valid output: a meeting with no decisions gives an empty decisions list. The web app offers the first five rows as downloads.

## Setup

Requires Python 3.10 or newer (developed on 3.14) and about 3 GB of free disk space for the Whisper model. ffmpeg is bundled through `imageio-ffmpeg`, so you do not need to install it.

1. Create an environment and install the dependencies.

   ```bash
   python -m venv .venv
   ```

   ```bash
   .venv/Scripts/python -m pip install -r requirements.txt
   ```

   (On macOS or Linux use `.venv/bin/python` in these commands.)

2. Get a free Groq API key at [console.groq.com](https://console.groq.com) (API Keys, Create API Key).
3. Copy `.env.example` to `.env` and paste the key after `GROQ_API_KEY=`. The `.env` file is git-ignored; never commit it.
4. The first transcription downloads the Whisper `large-v3` model (about 3 GB) and needs internet. It is cached afterwards.

### Optional settings (in `.env`)

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `tiny`, `base`, `small`, `medium`, `distil-large-v3` (faster, slightly less accurate) |
| `WHISPER_DEVICE` | `cpu` | `cuda` if you have an NVIDIA GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` for GPU |
| `WHISPER_CPU_THREADS` | your core count, max 16 | CPU threads for Whisper |
| `MINUTES_MODEL` | `qwen/qwen3.8-27b` | Groq model for stage 3 |

## Run

**Web interface**

```bash
.venv/Scripts/python -m streamlit run app.py
```

Open http://localhost:8501, upload a recording, click **Process recording**, then browse the tabs (Summary, Transcripts, Minutes, Decisions, Action items, Warnings, Downloads).

**Command line**

```bash
.venv/Scripts/python -m scripts.run_pipeline samples/test_meeting.wav
```


## Project structure

```
app.py                      Streamlit web interface
pipeline/
  errors.py                 PipelineError: messages written for end users
  config.py                 model names, settings, API key loading
  audio_tools.py            ffmpeg wrapper (decode, duration, loudness)
  validate.py               stage 0: input checks
  transcribe.py             stage 1: local Whisper
  refine.py                 stage 2: LLM #1, retries, change highlighting
  checks.py                 stage 2: numbers / negations / change-size checks
  minutes.py                stage 3: LLM #2, JSON validation, owner/deadline checks
  schema.py                 one record object shared by every output
  export.py                 Markdown + JSON files
  run.py                    runs stages 0 to 3 in order, reports status
prompts/
  refine_prompt.txt         instructions for LLM #1
  minutes_prompt.txt        instructions for LLM #2
scripts/run_pipeline.py     command-line entry point
samples/                    sample recording
```


## Known limitations

- **Names:** the refiner is told never to change names, but code checks cannot tell a misheard name from a misheard technical term, so this protection comes from the prompt alone.
- **No speaker labels:** an owner is reported only when a name is actually spoken. "I will do it" with no name gives `Unspecified`.
- **Decisions are a judgement call:** the model decides what counts as clearly agreed. Quote and hedging warnings help, but review them.
- **Speed:** local Whisper on a CPU can be slower than real time on long recordings. Use a smaller model or a GPU to speed it up.
- **Free-tier limits:** the two LLM stages share one Groq key and its rate limits. Calls are retried, but a long meeting may wait.
- **English only.**
