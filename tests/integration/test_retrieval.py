import numpy as np

from src.ingestion import Chunk, ChunkMetadata
from src.retrieval import HybridRetriever


class FakeEmbeddings:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, **_kwargs):
        return np.array([[float(len(text)), float(text.lower().count("sick"))] for text in texts])


def _chunks():
    return [
        Chunk("Sick leave requires a medical certificate.", ChunkMetadata(
            section_num=2, section_title="LEAVE", subsection_num="2.4", subsection_title="Policies",
            policy_name="Sick Leave", chunk_type="policy", split_group_id="sick", split_part=1, split_total=2,
        )),
        Chunk("Sick leave continuation details.", ChunkMetadata(
            section_num=2, section_title="LEAVE", subsection_num="2.4", subsection_title="Policies",
            policy_name="Sick Leave", chunk_type="policy", split_group_id="sick", split_part=2, split_total=2,
        )),
        Chunk("Recruitment vacancy process.", ChunkMetadata(
            section_num=1, section_title="RECRUITMENT", subsection_num="1.5", subsection_title="Procedures",
            procedure_code="ML-HR-01.P03", chunk_type="procedure",
        )),
    ]


def test_hybrid_retrieval_and_structural_expansion_use_temporary_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.SentenceTransformer", FakeEmbeddings)
    retriever = HybridRetriever(chroma_db_path=str(tmp_path / "chroma"), top_k=2, retrieval_candidates=3)
    retriever.load_or_create_collection("test_policies")
    retriever.add_chunks(_chunks())

    result = retriever.retrieve("sick leave")
    assert result
    assert any("Sick leave" in item["text"] for item in result)

    class Analysis:
        intent = "structural"
        procedure_code = None
        policy_name = "Sick Leave"
        section_num = 2
        subsection_num = "2.4"

    structural = retriever.retrieve("sick leave policy", analysis=Analysis())
    assert len(structural) == 2
    assert {item["metadata"].split_group_id for item in structural} == {"sick"}


def test_rrf_favors_documents_found_by_both_rankers():
    retriever = object.__new__(HybridRetriever)
    scores = retriever._reciprocal_rank_fusion(
        [("chunk_0", 0.1), ("chunk_1", 0.2)],
        [("chunk_1", 5.0), ("chunk_2", 2.0)],
        k=60,
    )
    assert scores["chunk_1"] > scores["chunk_0"]
    assert scores["chunk_1"] > scores["chunk_2"]
