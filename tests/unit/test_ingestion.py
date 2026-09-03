import json
from pathlib import Path

from src.ingestion import (
    Chunk, ChunkMetadata, PolicyManualParser, HierarchicalPDFParser,
    HR_DOC_CONFIG, EVENTS_DOC_CONFIG, COMMS_DOC_CONFIG, detect_doc_config
)
from src.retrieval import load_chunks_from_json


def test_chunk_serialization_preserves_metadata(tmp_path):
    chunk = Chunk(
        text="Policy text",
        metadata=ChunkMetadata(
            doc_id="ML-HR-01",
            doc_type="policy_manual",
            section_num=2,
            content_type="narrative",
            has_list=False,
            split_group_id="group",
            split_part=1,
            split_total=2,
        ),
    )
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([chunk.to_dict()]), encoding="utf-8")

    loaded = load_chunks_from_json(str(path))[0]
    assert loaded.text == chunk.text
    assert loaded.metadata.doc_id == "ML-HR-01"
    assert loaded.metadata.section_num == 2
    assert loaded.metadata.split_group_id == "group"
    assert loaded.metadata.split_total == 2


def test_large_chunk_creates_split_group_metadata():
    parser = object.__new__(PolicyManualParser)
    parser.doc_config = HR_DOC_CONFIG
    chunk = Chunk(
        text="Sentence. " * 300,
        metadata=ChunkMetadata(doc_id="ML-HR-01", section_num=2, chunk_type="policy")
    )

    parts = parser._split_large_chunk(chunk, max_size=100, overlap=10)

    assert len(parts) > 1
    assert {part.metadata.split_group_id for part in parts} == {parts[0].metadata.split_group_id}
    assert [part.metadata.split_part for part in parts] == list(range(1, len(parts) + 1))
    assert all(part.metadata.split_total == len(parts) for part in parts)
    assert all(part.metadata.doc_id == "ML-HR-01" for part in parts)


def test_detect_doc_config():
    assert detect_doc_config(Path("Events Management Policies and Procedures Manual (ML-ORG-01, V.01).pdf")) == EVENTS_DOC_CONFIG
    assert detect_doc_config(Path("Internal Communication Guideline (GL-ORG-01, V.01).pdf")) == COMMS_DOC_CONFIG
    assert detect_doc_config(Path("Human Resources Policies & Procedures Manual (ML-HR-01, V.01).docx.pdf")) == HR_DOC_CONFIG
    assert detect_doc_config(Path("Unknown_Document.pdf")) is None

