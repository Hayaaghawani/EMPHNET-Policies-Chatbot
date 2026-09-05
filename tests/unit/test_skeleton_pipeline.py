import pytest

from src.skeleton_pipeline import SkeletonValidationError, enrich_tree


def test_enrich_tree_extracts_unique_leaf_boundary():
    skeleton = {
        "doc_id": "TEST",
        "nodes": [{
            "id": "a",
            "path": "A",
            "node_type": "leaf",
            "heading": "A",
            "next_heading": "B",
        }],
    }
    result = enrich_tree(skeleton, "A\nalpha text\nB\nbeta text")
    assert result["nodes"][0]["own_text"] == "alpha text"


def test_enrich_tree_rejects_missing_heading():
    skeleton = {
        "doc_id": "TEST",
        "nodes": [{
            "id": "a",
            "path": "A",
            "node_type": "leaf",
            "heading": "A",
            "next_heading": "B",
        }],
    }
    with pytest.raises(SkeletonValidationError, match="Node a"):
        enrich_tree(skeleton, "C\nfirst\nB\nsecond")