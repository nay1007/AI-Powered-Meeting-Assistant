"""Web interface for the meeting assistant.

Run:  .venv/Scripts/python -m streamlit run app.py
"""
import os
import re
import tempfile

import pandas as pd
import streamlit as st

from pipeline.config import MINUTES_MODEL, REFINE_MODEL, STT_MODEL
from pipeline.errors import PipelineError
from pipeline.refine import highlighted_html, word_changes
from pipeline.run import run_pipeline
from pipeline.validate import ALLOWED_EXTENSIONS

st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")

DOWNLOADS = [
    ("raw_transcript.txt", "Raw transcript (.txt)", "text/plain"),
    ("refined_transcript.txt", "Refined transcript (.txt)", "text/plain"),
    ("minutes.md", "Meeting minutes (Markdown)", "text/markdown"),
    ("minutes.json", "Meeting minutes (JSON)", "application/json"),
    ("key_decisions.md", "Key decisions (Markdown)", "text/markdown"),
    ("key_decisions.json", "Key decisions (JSON)", "application/json"),
    ("action_items.md", "Action items (Markdown)", "text/markdown"),
    ("action_items.json", "Action items (JSON)", "application/json"),
]


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_on_upload(upload) -> None:
    """Run the pipeline on the uploaded file and keep the result in session_state."""
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(upload.name)) or "recording"
    stem = os.path.splitext(safe_name)[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, safe_name)
        with open(path, "wb") as f:
            f.write(upload.getbuffer())
        with st.status("Processing your recording...", expanded=True) as status:
            try:
                result = run_pipeline(path, out_dir=os.path.join("outputs", stem),
                                      on_status=lambda msg: st.write(msg))
            except PipelineError as exc:
                status.update(label="Processing failed", state="error")
                st.session_state["error"] = str(exc)
                return
            status.update(label="Done", state="complete", expanded=False)
    st.session_state["result"] = {
        "record": result.record.to_dict(),
        "files": {name: read_text(p) for name, p in result.files.items()},
    }


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown(
        f"1. **Speech-to-text:** faster-whisper `{STT_MODEL}` (runs on this computer)\n"
        f"2. **Refinement (LLM #1):** `{REFINE_MODEL}`\n"
        f"3. **Minutes (LLM #2):** `{MINUTES_MODEL}`\n\n"
        "Each stage runs in order on the uploaded audio. Nothing is pre-written."
    )
    st.caption("Supported: " + ", ".join(sorted(ALLOWED_EXTENSIONS)))
    st.caption("The audio is transcribed on this computer and is never uploaded. The transcript text is "
               "sent to Groq (free tier), which may use it to improve its products, so avoid confidential meetings.")

# ------------------------------------------------------------------------ main
st.title("📝 Meeting Assistant")
st.write("Upload an English meeting recording to get a transcript, minutes, decisions and action items.")

upload = st.file_uploader("Meeting recording", help="Any file is accepted here; unsupported ones get a clear message.")
if st.button("Process recording", type="primary", disabled=upload is None):
    run_on_upload(upload)

if "error" in st.session_state:
    st.error(st.session_state["error"])

if "result" in st.session_state:
    rec = st.session_state["result"]["record"]
    files = st.session_state["result"]["files"]
    meta = rec["metadata"]

    fell_back = meta.get("refinement_parts_fell_back_to_raw", 0)
    if fell_back:
        st.warning(f"{fell_back} part(s) of the refinement were rejected by the safety checks "
                   "and kept as raw text. See Warnings.")
    if rec["warnings"]:
        st.info(f"{len(rec['warnings'])} warning(s): see the Warnings tab.")

    tabs = st.tabs(["Summary", "Transcripts", "Minutes", "Decisions", "Action items", "Warnings", "Downloads"])

    with tabs[0]:
        st.subheader("Summary")
        st.write(rec["summary"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"{meta.get('duration_seconds', 0):.0f} s")
        c2.metric("Decisions / proposals", f"{len(rec['decisions'])} / {len(rec['proposals'])}")
        c3.metric("Action items", len(rec["action_items"]))

    with tabs[1]:
        raw, refined = rec["raw_transcript"], rec["refined_transcript"]
        n_changes = len(word_changes(raw, refined))
        left, right = st.columns(2)
        left.subheader("Raw transcript")
        left.write(raw)
        right.subheader("Refined transcript")
        right.markdown(highlighted_html(raw, refined), unsafe_allow_html=True)
        if n_changes:
            st.caption(f"{n_changes} correction(s) highlighted in the refined transcript. "
                       "Hover over a highlighted word to see the original.")
        else:
            st.caption("The refiner found nothing to correct.")

    with tabs[2]:
        if not rec["minutes"]:
            st.write("No minutes.")
        for topic in rec["minutes"]:
            st.markdown(f"**{topic['topic']}**")
            st.markdown("\n".join(f"- {p}" for p in topic["points"]))

    with tabs[3]:
        st.subheader("Key decisions (agreed)")
        if not rec["decisions"]:
            st.write("No decisions were reached.")
        for i, d in enumerate(rec["decisions"], 1):
            st.markdown(f"{i}. {d['decision']}")
            st.caption(f"“{d['evidence']}”")
        st.subheader("Proposals (not agreed)")
        if not rec["proposals"]:
            st.write("No open proposals.")
        for i, p in enumerate(rec["proposals"], 1):
            st.markdown(f"{i}. {p['proposal']}")
            st.caption(f"“{p['evidence']}”")

    with tabs[4]:
        if rec["action_items"]:
            df = pd.DataFrame(rec["action_items"])[["task", "owner", "deadline", "evidence"]]
            df.columns = ["Task", "Owner", "Deadline", "Evidence"]
            st.dataframe(df, hide_index=True)
        else:
            st.write("No action items were assigned.")

    with tabs[5]:
        for w in rec["warnings"]:
            st.warning(w)
        if not rec["warnings"]:
            st.success("No warnings.")

    with tabs[6]:
        for name, label, mime in DOWNLOADS:
            if name in files:
                st.download_button(label, files[name], file_name=name, mime=mime, key=name)
