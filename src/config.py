"""Runtime configuration for the EMPHNET policies chatbot.

Environment variables remain the deployment interface.  Keeping their parsing here
prevents the UI, indexing scripts, and RAG services from silently using different
defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class Settings:
    chroma_db_path: Path
    collection_name: str
    embedding_model: str
    embedding_device: str
    top_k_results: int
    rrf_k: int
    retrieval_candidates: int
    enable_reranking: bool
    reranker_model: str
    ollama_host: str
    ollama_model: str
    ollama_timeout: int
    hf_api_key: str | None
    hf_model: str
    llm_max_chunks: int
    llm_max_chars_per_chunk: int
    ollama_num_predict: int
    llm_list_max_chunks: int
    llm_list_max_chars_per_chunk: int
    list_num_predict: int
    pdf_path: Path
    chunks_path: Path
    pdf_dir: Path = Path("data/pdf")
    chunks_dir: Path = Path("data/chunks")
    index_schema_version: str = "multi-doc-v2"


def get_settings(*, load_environment: bool = True) -> Settings:
    """Read and validate the chatbot settings from the environment."""
    if load_environment:
        load_dotenv()

    legacy_list_predict = os.getenv("OLLAMA_LIST_NUM_PREDICT")
    list_predict_default = int(legacy_list_predict) if legacy_list_predict else 900
    return Settings(
        chroma_db_path=Path(os.getenv("CHROMA_DB_PATH", "chroma_storage")),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "emphnet_policies"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        top_k_results=_int("TOP_K_RESULTS", 5),
        rrf_k=_int("RRF_K", 60),
        retrieval_candidates=_int("RETRIEVAL_CANDIDATES", 20),
        enable_reranking=_bool("ENABLE_RERANKING", False),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        ollama_timeout=_int("OLLAMA_TIMEOUT", 300),
        hf_api_key=os.getenv("HF_API_KEY") or None,
        hf_model=os.getenv("HF_MODEL", "Qwen/Qwen3-32B"),
        llm_max_chunks=_int("LLM_MAX_CHUNKS", 3),
        llm_max_chars_per_chunk=_int("LLM_MAX_CHARS_PER_CHUNK", 1000),
        ollama_num_predict=_int("OLLAMA_NUM_PREDICT", 400),
        llm_list_max_chunks=_int("LLM_LIST_MAX_CHUNKS", 6),
        llm_list_max_chars_per_chunk=_int("LLM_LIST_MAX_CHARS", 6000),
        list_num_predict=_int("LIST_NUM_PREDICT", list_predict_default),
        pdf_path=Path(os.getenv(
            "PDF_PATH",
            "data/pdf/Human Resources Policies & Procedures Manual (ML-HR-01, V.01).docx.pdf",
        )),
        chunks_path=Path(os.getenv("CHUNKS_PATH", "data/chunks_inspection.json")),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdf")),
        chunks_dir=Path(os.getenv("CHUNKS_DIR", "data/chunks")),
    )

