import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from utils.dataset_loader import load_local_qmsum_sample, load_synthetic_sample, parse_uploaded_file
from utils.audio_tools import transcribe_audio_file
from utils.nlp import (
    build_chunks,
    extract_decisions,
    extract_action_items,
    extract_risks,
    generate_answer_local,
    generate_structured_summary_local,
    get_embedding_model,
    get_embeddings,
    search_chunks,
)
from utils.llm import llm_answer, llm_structured_summary

st.set_page_config(page_title="Meeting Intelligence Copilot", layout="wide", page_icon="🎙️")

CUSTOM_CSS = """
<style>
.main .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {
    padding: 1.25rem 1.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
    color: white;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem;
}
.small-muted {color: #64748b; font-size: 0.9rem;}
.answer-box {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 1rem 1rem 0.5rem 1rem;
    border-radius: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">🎙️ Meeting Intelligence Copilot</h1>
        <p style="margin:0.5rem 0 0 0; font-size:1.05rem;">
            Upload a meeting transcript or audio file, or load a public QMSum benchmark sample.
            Then extract summaries, decisions, action items, risks, and ask natural-language questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "meeting_doc" not in st.session_state:
    st.session_state.meeting_doc = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "source_label" not in st.session_state:
    st.session_state.source_label = None
if "processed" not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("Options")
    top_k = st.slider("Top transcript chunks", 3, 10, 5)
    use_llm = st.checkbox("Use OpenAI for richer summaries and answers", value=False)
    openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Optional. The app works without this using local retrieval + heuristics.",
    )
    st.markdown("---")
    st.caption("Without an API key, the app still supports transcript search, action-item extraction, decision extraction, and local summaries.")

source_tab, analytics_tab, qa_tab, transcript_tab = st.tabs(["Source", "Insights", "Ask", "Transcript"])

with source_tab:
    st.subheader("1) Choose the meeting source")
    source_mode = st.radio(
        "Input type",
        [
            "Use included synthetic sample",
            "Use public QMSum sample from local file",
            "Upload transcript (.txt, .md, .json)",
            "Upload audio (.wav, .mp3, .m4a)",
        ],
        horizontal=False,
    )

    uploaded_file = None
    qmsum_path = Path("data/qmsum_sample.json")

    if source_mode == "Upload transcript (.txt, .md, .json)":
        uploaded_file = st.file_uploader("Upload transcript file", type=["txt", "md", "json"])
    elif source_mode == "Upload audio (.wav, .mp3, .m4a)":
        uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
        st.info("Audio transcription uses Whisper locally. The first run may take longer while the model downloads.")
    elif source_mode == "Use public QMSum sample from local file":
        if qmsum_path.exists():
            st.success("Found local QMSum sample in data/qmsum_sample.json")
        else:
            st.warning(
                "No local QMSum sample found yet. Run `python scripts/download_qmsum_sample.py` to fetch one public MIT-licensed sample from the official QMSum repository."
            )

    process_clicked = st.button("Process Meeting", use_container_width=True)

    if process_clicked:
        try:
            if source_mode == "Use included synthetic sample":
                meeting_doc = load_synthetic_sample()
                source_label = "Included synthetic demo sample"
            elif source_mode == "Use public QMSum sample from local file":
                if not qmsum_path.exists():
                    raise FileNotFoundError(
                        "QMSum sample not found. Run `python scripts/download_qmsum_sample.py` first."
                    )
                meeting_doc = load_local_qmsum_sample(str(qmsum_path))
                source_label = "Public QMSum sample (MIT licensed source)"
            elif source_mode == "Upload transcript (.txt, .md, .json)":
                if uploaded_file is None:
                    raise ValueError("Please upload a transcript file.")
                meeting_doc = parse_uploaded_file(uploaded_file)
                source_label = f"Uploaded transcript: {uploaded_file.name}"
            else:
                if uploaded_file is None:
                    raise ValueError("Please upload an audio file.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                transcript_text = transcribe_audio_file(tmp_path)
                meeting_doc = {
                    "title": Path(uploaded_file.name).stem,
                    "source": "audio_upload",
                    "transcript_text": transcript_text,
                    "utterances": [{"speaker": "Transcript", "content": transcript_text}],
                    "reference_queries": [],
                }
                source_label = f"Uploaded audio: {uploaded_file.name}"

            chunks = build_chunks(meeting_doc["transcript_text"])
            model = get_embedding_model()
            embeddings = get_embeddings(model, [c["text"] for c in chunks])

            st.session_state.meeting_doc = meeting_doc
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.source_label = source_label
            st.session_state.processed = True
            st.success(f"Meeting processed successfully. Indexed {len(chunks)} transcript chunks.")
        except Exception as exc:
            st.exception(exc)

    if st.session_state.processed and st.session_state.meeting_doc:
        doc = st.session_state.meeting_doc
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><b>Source</b><br><span class='small-muted'>{st.session_state.source_label}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><b>Transcript lines</b><br><span class='small-muted'>{len(doc.get('utterances', []))}</span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><b>Transcript words</b><br><span class='small-muted'>{len(doc['transcript_text'].split()):,}</span></div>", unsafe_allow_html=True)

with analytics_tab:
    st.subheader("2) Structured meeting insights")
    if not st.session_state.processed:
        st.info("Process a meeting first from the Source tab.")
    else:
        doc = st.session_state.meeting_doc
        transcript_text = doc["transcript_text"]
        decisions = extract_decisions(transcript_text)
        action_items = extract_action_items(transcript_text)
        risks = extract_risks(transcript_text)

        if use_llm and openai_api_key:
            summary = llm_structured_summary(transcript_text, openai_api_key)
        else:
            summary = generate_structured_summary_local(transcript_text, decisions, action_items, risks)

        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.markdown("### Executive summary")
            st.markdown(f"<div class='answer-box'>{summary.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("### Quick stats")
            st.metric("Decisions found", len(decisions))
            st.metric("Action items found", len(action_items))
            st.metric("Risks / blockers found", len(risks))

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("### Decisions")
            if decisions:
                for item in decisions:
                    st.write(f"- {item}")
            else:
                st.caption("No clear decision phrases found.")
        with d2:
            st.markdown("### Action items")
            if action_items:
                for item in action_items:
                    st.write(f"- {item}")
            else:
                st.caption("No explicit action items found.")
        with d3:
            st.markdown("### Risks / blockers")
            if risks:
                for item in risks:
                    st.write(f"- {item}")
            else:
                st.caption("No explicit risks found.")

with qa_tab:
    st.subheader("3) Ask questions about the meeting")
    if not st.session_state.processed:
        st.info("Process a meeting first from the Source tab.")
    else:
        question = st.text_input(
            "Ask a question",
            placeholder="What were the key decisions? Who owns the API integration work? What risks were raised?",
        )
        ask_clicked = st.button("Ask", use_container_width=False)

        if ask_clicked:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                model = get_embedding_model()
                results = search_chunks(
                    question,
                    model,
                    st.session_state.embeddings,
                    st.session_state.chunks,
                    top_k=top_k,
                )
                if use_llm and openai_api_key:
                    answer = llm_answer(question, results, openai_api_key)
                else:
                    answer = generate_answer_local(question, results)

                st.markdown("### Answer")
                st.markdown(f"<div class='answer-box'>{answer.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

                st.markdown("### Supporting transcript snippets")
                for i, item in enumerate(results, start=1):
                    with st.expander(f"{i}. Chunk {item['chunk_id']} · score {item['score']:.3f}", expanded=(i == 1)):
                        st.write(item["text"])

        st.markdown("### Suggested questions")
        suggestions = [
            "What are the key decisions?",
            "List the action items and owners.",
            "What risks or blockers were mentioned?",
            "What timeline did the team agree on?",
        ]
        for s in suggestions:
            st.caption(f"• {s}")

with transcript_tab:
    st.subheader("4) Explore the transcript")
    if not st.session_state.processed:
        st.info("Process a meeting first from the Source tab.")
    else:
        doc = st.session_state.meeting_doc
        st.markdown(f"**Meeting title:** {doc.get('title', 'Untitled meeting')}  ")
        st.markdown(f"**Source:** {st.session_state.source_label}")

        if doc.get("reference_queries"):
            with st.expander("Benchmark reference queries from dataset"):
                for item in doc["reference_queries"][:10]:
                    st.write(f"**Q:** {item['query']}")
                    st.write(f"**A:** {item['answer']}")
                    st.markdown("---")

        st.markdown("### Transcript")
        for utt in doc.get("utterances", [])[:200]:
            st.write(f"**{utt.get('speaker', 'Speaker')}:** {utt.get('content', '')}")
