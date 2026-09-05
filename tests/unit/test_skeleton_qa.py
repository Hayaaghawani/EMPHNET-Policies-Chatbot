import json

from src.skeleton_pipeline import SkeletonCorpus, SkeletonQA
from src.skeleton_pipeline_types import Navigation


class FakeRetriever:
    def search(self, question):
        return [{"doc_id": "TEST", "id": "leaf", "score": 0.8}]


class FakeLLM:
    def navigate(self, question, outline, similarity_candidates):
        assert similarity_candidates
        return Navigation([("TEST", "leaf")], False)

    def answer(self, question, contexts):
        assert contexts[0]["text"] == "Grounded policy text"
        return {"answer": "Grounded answer", "sources": contexts}


def test_qa_merges_retrieval_and_navigation_context(tmp_path):
    tree_dir = tmp_path / "trees"
    tree_dir.mkdir()
    (tree_dir / "TEST.json").write_text(json.dumps({
        "doc_id": "TEST",
        "nodes": [{
            "id": "leaf",
            "path": "Policy > Leaf",
            "node_type": "leaf",
            "own_text": "Grounded policy text",
        }],
    }), encoding="utf-8")
    outline_path = tmp_path / "outline.json"
    outline_path.write_text(json.dumps([{
        "doc_id": "TEST", "id": "leaf", "path": "Policy > Leaf", "node_type": "leaf"
    }]), encoding="utf-8")

    qa = SkeletonQA(SkeletonCorpus(tree_dir, outline_path), FakeLLM(), FakeRetriever())
    result = qa.answer("What is the policy?")

    assert result["answer"] == "Grounded answer"
    assert result["sources"][0]["path"] == "Policy > Leaf"
    assert result["retrieval"][0]["id"] == "leaf"
