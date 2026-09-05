"""Streamlit UI for the skeleton-driven EMPHNET policy assistant."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.skeleton_generation import SkeletonLLM
from src.skeleton_pipeline import SkeletonCorpus, SkeletonQA
from src.skeleton_retrieval import SkeletonHybridRetriever

load_dotenv()
st.set_page_config(page_title="EMPHNET Policies Assistant", page_icon="P", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding-top: 3rem; }
    .answer { background: #f1f7f4; border-left: 4px solid #187a58; padding: 1rem 1.25rem; }
    .source { border-left: 3px solid #d68b32; padding: .5rem 1rem; margin: .5rem 0; }
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


st.title("EMPHNET Policies Assistant")
st.caption("HR, Events Management, and Internal Communication documents")
qa = initialize_qa()
if qa is None:
    st.error("Structured data is not built yet. Run `python build_skeleton_data.py` after correcting any skeleton validation errors.")
    st.stop()

question = st.text_area(
    "Question",
    placeholder="Ask a question in English or Arabic",
    height=110,
    label_visibility="collapsed",
)
if st.button("Get answer", type="primary", use_container_width=True) and question.strip():
    with st.spinner("Selecting the relevant policy section and drafting an answer..."):
        try:
            result = qa.answer(question.strip())
        except Exception as exc:
            st.error(f"The request could not be completed: {exc}")
        else:
            st.markdown("### Answer")
            st.markdown(f'<div class="answer">{result["answer"]}</div>', unsafe_allow_html=True)
            with st.expander(f"Sources ({len(result['sources'])})", expanded=True):
                for source in result["sources"]:
                    st.markdown(
                        f'<div class="source"><strong>{source["path"]}</strong><br>{source["text"]}</div>',
                        unsafe_allow_html=True,
                    )
elif not question.strip():
    st.info("Enter a question to begin.")
