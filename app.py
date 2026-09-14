"""Streamlit UI for the skeleton-driven EMPHNET policy assistant."""

import os
from pathlib import Path

import streamlit as st
st.set_page_config(page_title="EMPHNET Policies Assistant", page_icon="P", layout="wide")

LOGO_PATH = Path("assets/emphnet logo.png")

def render_header() -> None:
    logo_column, title_column = st.columns([0.8, 5], gap="small", vertical_alignment="center")
    with logo_column:
        st.image(str(LOGO_PATH), width=72)
    with title_column:
        st.title("EMPHNET Policies Assistant")

# Microsoft OIDC gate: no retrieval, model loading, or app UI runs for guests.
if not st.user.is_logged_in:
    render_header()
    st.write("Sign in with your EMPHNET Microsoft account to access the policy assistant.")
    if st.button("Log in with Microsoft", type="primary"):
        st.login("microsoft")
    st.stop()

with st.sidebar:
    st.write(f"Signed in as {st.user.email}")
    if st.button("Log out"):
        st.logout()

from dotenv import load_dotenv

from src.skeleton_generation import SkeletonLLM
from src.skeleton_pipeline import SkeletonCorpus, SkeletonQA
from src.skeleton_retrieval import SkeletonHybridRetriever

load_dotenv()
MAX_MEMORY_TURNS = 4

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "navigation_memory" not in st.session_state:
    st.session_state.navigation_memory = []

with st.sidebar:
    if st.button("New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.navigation_memory = []
        st.rerun()

st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding-top: 3rem; }
    .answer { background: #e8f3ed; border-left: 4px solid #187a58; padding: 1rem 1.25rem; color: #111111 !important; }
    .answer * { color: #111111 !important; }
    .source { background: #fff4e5; border-left: 3px solid #d68b32; padding: .5rem 1rem; margin: .5rem 0; color: #111111 !important; }
    .source * { color: #111111 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_qa() -> SkeletonQA | None:
    tree_dir = Path(os.getenv("ENRICHED_NODES_DIR", "data/enriched_nodes"))
    outline_path = Path(os.getenv("OUTLINE_PATH", "data/outline.json"))
    if not tree_dir.exists() or not outline_path.exists():
        return None
    corpus = SkeletonCorpus(tree_dir, outline_path)
    retriever = SkeletonHybridRetriever(
        tree_dir=tree_dir,
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        top_k=int(os.getenv("TOP_K_RESULTS", "5")),
    )
    llm = SkeletonLLM(
        hf_api_key=os.getenv("HF_API_KEY", ""),
        hf_model=os.getenv("HF_MODEL", "Qwen/Qwen3-32B"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
    )
    return SkeletonQA(corpus, llm, retriever)

render_header()
st.caption("HR, Events Management, and Internal Communication documents")
qa = initialize_qa()
if qa is None:
    st.error("Structured data is not built yet. Run `python build_skeleton_data.py` after correcting any skeleton validation errors.")
    st.stop()

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer">{turn["answer"]}</div>', unsafe_allow_html=True)
        with st.expander(f"Sources ({len(turn['sources'])})", expanded=False):
            for source in turn["sources"]:
                st.markdown(
                    f'<div class="source"><strong>{source["document"]}</strong><br><strong>{source["path"]}</strong><br>{source["text"]}</div>',
                    unsafe_allow_html=True,
                )

question = st.chat_input("Ask a question in English or Arabic")
if question and question.strip():
    with st.spinner("Selecting the relevant policy section and drafting an answer..."):
        try:
            result = qa.answer(question.strip(), memory=st.session_state.navigation_memory)
        except Exception as exc:
            st.error(f"The request could not be completed: {exc}")
        else:
            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": result["answer"],
                "sources": result["sources"],
            })
            st.session_state.navigation_memory.append({
                "question": question.strip(),
                "nodes": qa.corpus.memory_nodes(result["navigation"]),
            })
            st.session_state.navigation_memory = st.session_state.navigation_memory[-MAX_MEMORY_TURNS:]
            st.rerun()
