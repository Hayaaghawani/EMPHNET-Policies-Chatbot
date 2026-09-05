"""Hybrid retrieval over enriched skeleton nodes.

The index contains complete semantic leaves, forms, and generated table rows,
not arbitrary character chunks. BM25 handles exact policy vocabulary while
embeddings handle questions whose wording differs from a node heading.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06ff]+", text.casefold())


class SkeletonHybridRetriever:
    """Retrieve complete enriched nodes with BM25 plus embedding similarity."""

    def __init__(
        self,
        tree_dir: Path = Path("data/enriched_nodes"),
        embedding_model: str = "intfloat/multilingual-e5-large",
        embedding_device: str = "cpu",
        top_k: int = 5,
    ):
        self.tree_dir = tree_dir
        self.embedding_model_name = embedding_model
        self.embedding_device = embedding_device
        self.top_k = top_k
        self.records: list[dict[str, Any]] = []
        self._load_records()
        self.bm25 = BM25Okapi([_tokens(record["search_text"]) for record in self.records])
        logger.info("Loaded %s enriched nodes for hybrid retrieval", len(self.records))
        self.embeddings = SentenceTransformer(embedding_model, device=embedding_device)
        self.node_embeddings = self.embeddings.encode(
            [f"passage: {record['search_text']}" for record in self.records],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _load_records(self) -> None:
        for tree_path in sorted(self.tree_dir.glob("*.json")):
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            for node in tree.get("nodes", []):
                if not node.get("own_text"):
                    continue
                self.records.append({
                    "doc_id": tree["doc_id"],
                    "id": node["id"],
                    "path": node["path"],
                    "node_type": node["node_type"],
                    "search_text": f"{node['path']}\n{node['own_text']}",
                })

    def search(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.top_k
        query_embedding = self.embeddings.encode(
            [f"query: {question}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        semantic = np.asarray(self.node_embeddings) @ np.asarray(query_embedding)
        lexical = np.asarray(self.bm25.get_scores(_tokens(question)), dtype=float)
        if lexical.max(initial=0) > 0:
            lexical = lexical / lexical.max()
        combined = 0.7 * semantic + 0.3 * lexical
        indices = np.argsort(combined)[::-1][:limit]
        results = []
        for index in indices:
            record = dict(self.records[int(index)])
            record["semantic_score"] = float(semantic[index])
            record["lexical_score"] = float(lexical[index])
            record["score"] = float(combined[index])
            results.append(record)
        return results