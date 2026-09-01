"""Explicit, safe lifecycle operations for the Chroma policy index."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import chromadb

from .config import Settings
from .retrieval import HybridRetriever, load_chunks_from_json


@dataclass(frozen=True)
class IndexStatus:
    exists: bool
    collection_name: str
    document_count: int
    fingerprint: str | None
    is_current: bool | None
    metadata: dict[str, Any]


class IndexManager:
    """Build and inspect an index without allowing implicit destructive changes."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(path=str(self.settings.chroma_db_path))

    def chunk_fingerprint(self) -> str | None:
        if not self.settings.chunks_path.exists():
            return None
        return hashlib.sha256(self.settings.chunks_path.read_bytes()).hexdigest()

    def verify(self) -> IndexStatus:
        """Inspect the configured collection without changing it."""
        client = self._client()
        try:
            collection = client.get_collection(self.settings.collection_name)
        except Exception as exc:
            if _is_missing_collection_error(exc):
                return IndexStatus(False, self.settings.collection_name, 0, None, None, {})
            raise

        metadata = dict(collection.metadata or {})
        fingerprint = metadata.get("chunks_fingerprint")
        current = self.chunk_fingerprint()
        return IndexStatus(
            exists=True,
            collection_name=self.settings.collection_name,
            document_count=collection.count(),
            fingerprint=fingerprint,
            is_current=(fingerprint == current) if fingerprint and current else None,
            metadata=metadata,
        )

    def build(self, *, rebuild: bool = False, confirm_rebuild: bool = False) -> IndexStatus:
        """Build from chunks; replacing an index needs two explicit flags."""
        if not self.settings.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.settings.chunks_path}")

        status = self.verify()
        if status.exists and status.document_count:
            if not rebuild:
                raise RuntimeError("Index already exists. Use --rebuild --confirm-rebuild to replace it.")
            if not confirm_rebuild:
                raise RuntimeError("Refusing to rebuild without --confirm-rebuild.")
            self._client().delete_collection(self.settings.collection_name)

        chunks = load_chunks_from_json(str(self.settings.chunks_path))
        retriever = HybridRetriever(
            chroma_db_path=str(self.settings.chroma_db_path),
            embedding_model=self.settings.embedding_model,
            embedding_device=self.settings.embedding_device,
            top_k=self.settings.top_k_results,
            rrf_k=self.settings.rrf_k,
            retrieval_candidates=self.settings.retrieval_candidates,
        )
        retriever.load_or_create_collection(self.settings.collection_name)
        retriever.add_chunks(chunks)
        retriever.collection.modify(metadata={
            "hnsw:space": "cosine",
            "chunks_fingerprint": self.chunk_fingerprint() or "",
            "embedding_model": self.settings.embedding_model,
            "index_schema_version": self.settings.index_schema_version,
        })
        retriever.persist()
        return self.verify()

    def create_retriever(self) -> HybridRetriever:
        """Create a retriever for an existing index; never builds it implicitly."""
        status = self.verify()
        if not status.exists or not status.document_count:
            raise RuntimeError(
                "The configured index is unavailable or empty. Build it explicitly with embed_chunks.py."
            )
        retriever = HybridRetriever(
            chroma_db_path=str(self.settings.chroma_db_path),
            embedding_model=self.settings.embedding_model,
            embedding_device=self.settings.embedding_device,
            top_k=self.settings.top_k_results,
            rrf_k=self.settings.rrf_k,
            retrieval_candidates=self.settings.retrieval_candidates,
        )
        retriever.load_or_create_collection(self.settings.collection_name)
        return retriever


def _is_missing_collection_error(exc: Exception) -> bool:
    """Recognize version-specific Chroma not-found errors without masking others."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "notfound" in name or "does not exist" in message or "not found" in message
