#!/usr/bin/env python3
"""
Embeddings Pipeline — Load per-document chunks and embed them into ChromaDB.

Deletes and recreates the collection so embeddings always match current chunks.

Per-doc chunk files are read from data/chunks/*.json (one per document).
Falls back to data/chunks_inspection.json if the chunks/ directory doesn't exist.
"""
import logging
from pathlib import Path

import chromadb

from src.retrieval import HybridRetriever, load_chunks_from_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "emphnet_policies"
CHROMA_PATH = Path("chroma_storage")


def main():
    logger.info("=" * 60)
    logger.info("EMBEDDINGS PIPELINE — MULTI-DOCUMENT")
    logger.info("=" * 60)

    # Discover per-doc chunk files
    chunks_dir = Path("data/chunks")
    legacy_file = Path("data/chunks_inspection.json")

    chunk_files = []
    if chunks_dir.exists():
        chunk_files = sorted(chunks_dir.glob("*_chunks.json"))

    if not chunk_files:
        # Fall back to merged legacy file
        if legacy_file.exists():
            logger.info(f"No per-doc files in {chunks_dir}, using legacy {legacy_file}")
            chunk_files = [legacy_file]
        else:
            logger.error("No chunk files found. Run: python src/ingestion.py")
            return

    # Load all chunks from all files
    all_chunks = []
    for f in chunk_files:
        doc_chunks = load_chunks_from_json(str(f))
        logger.info(f"  {f.name}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    logger.info(f"Total: {len(all_chunks)} chunks across {len(chunk_files)} document(s)")

    # Drop and recreate collection
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        logger.info("No existing collection to delete")

    retriever = HybridRetriever(chroma_db_path=str(CHROMA_PATH))
    retriever.collection = retriever.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    retriever.add_chunks(all_chunks)



    logger.info("=" * 60)
    logger.info("EMBEDDINGS COMPLETE")
    logger.info(f"Vector database: {CHROMA_PATH}")
    logger.info(f"Collection:      {COLLECTION_NAME}")
    logger.info(f"Total embedded:  {len(all_chunks)} chunks")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

