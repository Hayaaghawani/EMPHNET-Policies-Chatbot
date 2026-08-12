"""
Hybrid Retrieval Module - Vector Search + BM25

Combines semantic similarity (vector embeddings) with keyword matching (BM25)
for improved retrieval accuracy on policy documents.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

from .ingestion import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Vector Search (semantic similarity via sentence-transformers)
    - BM25 Search (keyword-based ranking)
    
    Formula: hybrid_score = 0.8 * vector_score + 0.2 * bm25_score
    """
    
    def __init__(
        self,
        chroma_db_path: str = "chroma_storage",
        embedding_model: str = "intfloat/multilingual-e5-large",
        embedding_device: str = "cpu",
        top_k: int = 8
    ):
        """
        Initialize hybrid retriever
        
        Args:
            chroma_db_path: Path to ChromaDB persistent storage
            embedding_model: HuggingFace model ID for embeddings
            embedding_device: "cpu" or "cuda"
            top_k: Number of chunks to retrieve
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.embedding_model_name = embedding_model
        self.device = embedding_device
        self.top_k = top_k
        
        # Initialize embeddings
        logger.info(f"Loading embeddings model: {embedding_model}")
        self.embeddings = SentenceTransformer(embedding_model, device=self.device)
        
        # Initialize ChromaDB
        self.chroma_client = self._init_chroma()
        self.collection = None
        
        # BM25 index (built from collection)
        self.bm25_index = None
        self.chunks_list = []
    
    def _init_chroma(self) -> chromadb.Client:
        """Initialize ChromaDB with persistent storage using new API"""
        # Ensure directory exists
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Use new ChromaDB persistent client API
        client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        return client
    
    def load_or_create_collection(self, collection_name: str = "emphnet_policies") -> None:
        """
        Load existing ChromaDB collection or create new one
        
        Args:
            collection_name: Name of the collection
        """
        # Try to get existing collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
            # Rebuild BM25 index from existing collection
            self._rebuild_bm25_from_collection()
        except:
            # Create new collection with custom embedding function
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None  # We'll embed manually
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to vector store and build BM25 index
        
        Args:
            chunks: List of Chunk objects with text and metadata
        """
        if not self.collection:
            raise ValueError("Collection not loaded. Call load_or_create_collection first.")
        
        if not chunks:
            logger.warning("No chunks provided")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to collection")
        
        # Store chunks for BM25
        self.chunks_list = chunks
        
        # Generate embeddings (E5 models require "passage:" prefix)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.encode(
            [self._format_passage(t) for t in texts],
            show_progress_bar=True
        )
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        # Convert metadata to simple types (strings/ints/floats only - ChromaDB limitation)
        metadatas = []
        for chunk in chunks:
            meta = chunk.metadata if isinstance(chunk.metadata, dict) else chunk.metadata.__dict__
            # Filter to only simple types, convert None to empty string
            filtered_meta = {}
            for k, v in meta.items():
                if v is None:
                    filtered_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    filtered_meta[k] = v
                else:
                    filtered_meta[k] = str(v)
            metadatas.append(filtered_meta)
        documents = texts
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
        
        # Build BM25 index
        self._build_bm25_index()
        logger.info("BM25 index built")
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from chunks"""
        # Tokenize all chunks
        corpus = [chunk.text.lower().split() for chunk in self.chunks_list]
        self.bm25_index = BM25Okapi(corpus)
    
    def _rebuild_bm25_from_collection(self) -> None:
        """Rebuild BM25 index from existing collection (used when loading from disk)"""
        if not self.collection:
            logger.warning("Collection not loaded, cannot rebuild BM25")
            return
        
        try:
            # Get all documents from the collection
            all_docs = self.collection.get(include=["documents", "metadatas"])
            if not all_docs or not all_docs.get("documents"):
                logger.warning("No documents found in collection")
                return
            
            documents = all_docs.get("documents", [])
            metadatas = all_docs.get("metadatas", [])
            self.chunks_list = []
            for doc, meta in zip(documents, metadatas):
                metadata = ChunkMetadata(**{k: (v if v != "" else None) for k, v in meta.items()})
                self.chunks_list.append(Chunk(text=doc, metadata=metadata))
            
            # Build BM25 index
            corpus = [doc.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"Rebuilt BM25 index from {len(documents)} documents in collection")
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
    
    @staticmethod
    def _format_query(query: str) -> str:
        """E5 embedding models require a query prefix."""
        return f"query: {query}"

    @staticmethod
    def _format_passage(text: str) -> str:
        """E5 embedding models require a passage prefix."""
        return f"passage: {text}"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: float = 0.8,
        bm25_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search
        
        Args:
            query: User question/query in English or Arabic
            top_k: Number of results (defaults to self.top_k)
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 ranking (0-1)
        
        Returns:
            List of dicts with keys:
            - text: chunk text
            - metadata: ChunkMetadata
            - score: combined relevance score (0-1)
            - source: formatted source string
        """
        if not self.collection or not self.bm25_index:
            raise ValueError("Retriever not initialized. Call add_chunks first.")
        
        top_k = top_k or self.top_k
        
        # Normalize weights
        total = vector_weight + bm25_weight
        vector_weight /= total
        bm25_weight /= total
        
        logger.debug(f"Retrieving top {top_k} chunks for query: {query[:100]}...")
        
        # Vector search (E5 models require "query:" prefix)
        query_embedding = self.embeddings.encode([self._format_query(query)])[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2  # Get more to re-rank
        )
        
        # Extract vector similarities (ChromaDB returns distances, convert to similarity)
        vector_scores = {}
        for i, (doc_id, distance) in enumerate(zip(
            vector_results['ids'][0],
            vector_results['distances'][0]
        )):
            # Convert distance to similarity (assuming cosine: distance in [0,2], similarity in [0,1])
            similarity = 1 - (distance / 2)
            vector_scores[doc_id] = similarity
        
        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores_raw = self.bm25_index.get_scores(query_tokens)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {
            f"chunk_{i}": score / max_bm25
            for i, score in enumerate(bm25_scores_raw)
        }
        
        # Combine scores
        combined_scores = {}
        for doc_id in vector_scores:
            combined_scores[doc_id] = (
                vector_weight * vector_scores.get(doc_id, 0) +
                bm25_weight * bm25_scores.get(doc_id, 0)
            )
        
        # Sort and return top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_results:
            chunk_idx = int(doc_id.split('_')[1])
            chunk = self.chunks_list[chunk_idx]
            
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": score,
                "source": self._format_source(chunk.metadata)
            })
        
        return results
    
    @staticmethod
    def _format_source(metadata) -> str:
        """Format metadata into human-readable source citation"""
        if isinstance(metadata, dict):
            meta = metadata
        else:
            meta = metadata.__dict__ if hasattr(metadata, '__dict__') else {}

        parts = [meta.get('document_title', 'HR Policies & Procedures Manual')]

        if meta.get('section_num'):
            parts.append(f"Section {meta['section_num']:02d}")

        if meta.get('policy_name'):
            parts.append(meta['policy_name'])

        if meta.get('procedure_code'):
            parts.append(meta['procedure_code'])

        if meta.get('subsection_num') and meta.get('subsection_title'):
            parts.append(f"{meta['subsection_num']} {meta['subsection_title']}")

        if meta.get('page_number'):
            parts.append(f"p. {meta['page_number']}")

        return " • ".join(parts)
    
    def persist(self) -> None:
        """Persist ChromaDB collection to disk"""
        logger.info(f"Collection is automatically persisted by ChromaDB to {self.chroma_db_path}")


def load_chunks_from_json(json_path: str) -> List[Chunk]:
    """Load chunks from JSON inspection file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    for item in data:
        metadata = ChunkMetadata(**item['metadata'])
        chunk = Chunk(text=item['text'], metadata=metadata)
        chunks.append(chunk)
    
    return chunks


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retriever
    retriever = HybridRetriever(
        chroma_db_path="chroma_storage",
        top_k=5
    )
    
    # Load or create collection
    retriever.load_or_create_collection("emphnet_policies")
    
    # Load chunks from inspection file
    chunks = load_chunks_from_json("data/chunks_inspection.json")
    
    # Add to retriever
    retriever.add_chunks(chunks)
    
    # Example retrieval
    query = "What is the policy on sick leave?"
    results = retriever.retrieve(query)
    
    print(f"\nRetrieved {len(results)} results for: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Text: {result['text'][:200]}...")
        print()
