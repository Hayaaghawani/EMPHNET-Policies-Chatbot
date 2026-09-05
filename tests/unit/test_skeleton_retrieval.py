import json

import numpy as np

from src.skeleton_retrieval import SkeletonHybridRetriever, _tokens


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([2.0, 0.0])


def test_tokens_support_english_and_arabic():
    assert _tokens("Sick leave إجازة") == ["sick", "leave", "إجازة"]


def test_search_combines_lexical_and_semantic_scores():
    retriever = object.__new__(SkeletonHybridRetriever)
    retriever.top_k = 2
    retriever.records = [
        {"doc_id": "A", "id": "one", "path": "One", "node_type": "leaf", "search_text": "one"},
        {"doc_id": "B", "id": "two", "path": "Two", "node_type": "leaf", "search_text": "two"},
    ]
    retriever.bm25 = FakeBM25()
    retriever.embeddings = type("Embeddings", (), {
        "encode": lambda self, texts, **kwargs: np.array([[1.0, 0.0]])
    })()
    retriever.node_embeddings = np.array([[0.8, 0.0], [0.2, 0.0]])

    results = retriever.search("question")

    assert results[0]["id"] == "one"
    assert results[0]["lexical_score"] == 1.0
    assert len(results) == 2


def test_load_records_uses_only_nodes_with_text(tmp_path):
    tree_dir = tmp_path / "trees"
    tree_dir.mkdir()
    (tree_dir / "A.json").write_text(json.dumps({
        "doc_id": "A",
        "nodes": [
            {"id": "leaf", "path": "Leaf", "node_type": "leaf", "own_text": "Useful text"},
            {"id": "parent", "path": "Parent", "node_type": "parent"},
        ],
    }), encoding="utf-8")

    retriever = object.__new__(SkeletonHybridRetriever)
    retriever.tree_dir = tree_dir
    retriever.records = []
    retriever._load_records()

    assert [(record["doc_id"], record["id"]) for record in retriever.records] == [("A", "leaf")]
