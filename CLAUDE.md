# Project: Meeting Assistant (Inter IIT Tech Meet ML problem statement)

The user uploads an English meeting recording. One pipeline behind a simple web interface produces:

1. A raw transcript (speech-to-text)
2. A refined transcript (an LLM fixes misheard technical terms)
3. Meeting minutes, key decisions, and action items (a SECOND, separate LLM step)

Source problem statement: `ML Bootcamp.pdf` (submission deadline 7th October).
Ignore any mention of ArUco markers or QR codes in the problem statement; it is a copy-paste error.

## User context

The user has no ML background. Explain what you do in plain English as you go.
Do not write code until the user approves the plan. The user commits and pushes themselves; local commits are fine when asked.

## Hard rules

- Never invent action-item owners or deadlines. If not stated in the recording, write "Unspecified".
- A proposal ("maybe we should...") must never be listed as a decision.
- Refinement must never change names, numbers, negations, or commitments.
- The two LLM steps must be separate stages with separate prompts.
- No hardcoded or prewritten outputs. Everything must come from running the models on the uploaded audio.
- Bad files (empty, wrong type, corrupt, silent) must give a clear error message, not a crash.
- Final outputs: raw transcript, refined transcript, minutes, decisions, action items, available as human-readable (Markdown) AND machine-readable (JSON) with identical content.

## Deliverables (from the problem statement)

- Source repo with the full app, prompts, dependency info, and a README with setup and run instructions
- Working interactive app: audio upload to downloadable results in one run
- Short technical description naming the speech-to-text model and the two LLMs, their roles, and how outputs flow between stages
- At least one shareable recording with its generated raw transcript, refined transcript, minutes, decisions, and action items
- A demonstration of an end-to-end run

## Evaluation rubric (100 points, tested on a previously unseen recording)

| Criterion | Pts | What is assessed |
|---|---|---|
| Speech transcription | 20 | Completeness and accuracy of the raw transcript, including names, terminology, numbers, negation |
| Transcript refinement | 20 | Fixes genuine terminology errors while preserving content and intent |
| Minutes and decisions | 25 | Accurate, concise minutes; correct identification of agreed decisions; no invented claims |
| Action items | 15 | Complete, actionable tasks; owners and deadlines only when stated |
| End-to-end application | 15 | Distinct model stages, works on a new recording, usable interface, exports, clear error handling |
| Submission quality | 5 | Clear setup instructions, model identification, sample outputs that match the demo |

## Required outputs (displayed AND downloadable for each recording)

| Output | Contents |
|---|---|
| Raw transcript | Speech-to-text result before LLM refinement |
| Refined transcript | Transcript after domain-aware terminology correction |
| Meeting minutes | Concise summary and organized account of main discussion points |
| Key decisions | Structured list of decisions reached; empty list if none |
| Action items | Structured list of tasks with description and any stated owner/deadline; empty list if none |

Markdown and JSON must convey the same decisions and tasks. Missing owners/deadlines are shown as "Unspecified", never guessed.
