#!/usr/bin/env python3
"""
Embeddings Pipeline - Load chunks and embed them into ChromaDB vector store.
Deletes and recreates the collection to ensure embeddings match current chunks.
"""
import json
import logging
from pathlib import Path

import chromadb

from src.retrieval import HybridRetriever, load_chunks_from_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    chunks_file = Path("data/chunks_inspection.json")
    chroma_path = Path("chroma_storage")

    logger.info("=" * 60)
    logger.info("EMBEDDINGS PIPELINE")
    logger.info("=" * 60)

    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.info("Run: python src/ingestion.py")
        return

    chunks = load_chunks_from_json(str(chunks_file))
    logger.info(f"Loaded {len(chunks)} chunks")

    # Drop existing collection (avoids Windows file-lock issues with rmtree)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection("emphnet_policies")
        logger.info("Deleted existing collection")
    except Exception:
        logger.info("No existing collection to delete")

    retriever = HybridRetriever(chroma_db_path=str(chroma_path))
    retriever.load_or_create_collection("emphnet_policies")
    retriever.add_chunks(chunks)

    logger.info("=" * 60)
    logger.info("EMBEDDINGS COMPLETE")
    logger.info(f"Vector database: {chroma_path}")
    logger.info(f"Total chunks embedded: {len(chunks)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
