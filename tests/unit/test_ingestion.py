import json

from src.ingestion import Chunk, ChunkMetadata, HierarchicalPDFParser
from src.retrieval import load_chunks_from_json


def test_chunk_serialization_preserves_metadata(tmp_path):
    chunk = Chunk(
        text="Policy text",
        metadata=ChunkMetadata(section_num=2, split_group_id="group", split_part=1, split_total=2),
    )
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([chunk.to_dict()]), encoding="utf-8")

    loaded = load_chunks_from_json(str(path))[0]
    assert loaded.text == chunk.text
    assert loaded.metadata.section_num == 2
    assert loaded.metadata.split_group_id == "group"
    assert loaded.metadata.split_total == 2


def test_large_chunk_creates_split_group_metadata():
    parser = object.__new__(HierarchicalPDFParser)
    chunk = Chunk(text="Sentence. " * 300, metadata=ChunkMetadata(section_num=2, chunk_type="policy"))

    parts = parser._split_large_chunk(chunk, max_size=100, overlap=10)

    assert len(parts) > 1
    assert {part.metadata.split_group_id for part in parts} == {parts[0].metadata.split_group_id}
    assert [part.metadata.split_part for part in parts] == list(range(1, len(parts) + 1))
    assert all(part.metadata.split_total == len(parts) for part in parts)
