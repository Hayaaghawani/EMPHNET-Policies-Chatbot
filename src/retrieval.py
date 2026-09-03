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
from .document_structure import DocumentStructureIndex, QueryAnalysis, analyze_query

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
        top_k: int = 5,
        retrieval_candidates: Optional[int] = None,
        rrf_k: int = 60,
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
        self.retrieval_candidates = retrieval_candidates or max(top_k * 2, 10)
        self.rrf_k = rrf_k
        
        # Initialize embeddings
        logger.info(f"Loading embeddings model: {embedding_model}")
        self.embeddings = SentenceTransformer(embedding_model, device=self.device)
        
        # Initialize ChromaDB
        self.chroma_client = self._init_chroma()
        self.collection = None
        
        # BM25 index (built from collection)
        self.bm25_index = None
        self.chunks_list = []
        self.structure_index: Optional[DocumentStructureIndex] = None
    
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

    def _chunk_id_for(self, idx: int) -> str:
        """Generate consistent unique chunk ID string for chunk index idx."""
        if 0 <= idx < len(self.chunks_list):
            meta = self._meta_dict(self.chunks_list[idx].metadata)
            doc_id = meta.get("doc_id") or "chunk"
            return f"{doc_id}_chunk_{idx}"
        return f"chunk_{idx}"

    @staticmethod
    def _chunk_idx_from_id(doc_id: str) -> int:
        """Extract integer chunk index from doc_id string (e.g. 'ML-HR-01_chunk_5' or 'chunk_5')."""
        return int(doc_id.rsplit("_", 1)[1])
    
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
        self.structure_index = DocumentStructureIndex(chunks)

        # Generate embeddings (E5 models require "passage:" prefix on embed only)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.encode(
            [self._format_passage(t) for t in texts],
            show_progress_bar=True,
        )
        
        # Prepare data for ChromaDB — IDs are globally unique across all docs
        ids = [self._chunk_id_for(i) for i in range(len(chunks))]
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
        
        # Batch upload to ChromaDB
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                documents=documents[i:end_idx]
            )
        
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
        
        # Build BM25 index from full chunk texts
        corpus = [chunk.text.lower().split() for chunk in self.chunks_list]
        self.bm25_index = BM25Okapi(corpus)
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

            # Only pass fields that ChunkMetadata actually accepts
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(ChunkMetadata)}

            self.chunks_list = []
            for doc, meta in zip(documents, metadatas):
                clean = {k: (v if v != "" else None) for k, v in meta.items()
                         if k in valid_fields}
                for int_key in ("section_num", "page_number", "split_part", "split_total"):
                    if clean.get(int_key) is not None:
                        clean[int_key] = int(clean[int_key])
                for bool_key in ("has_list",):
                    if bool_key in clean and clean[bool_key] is not None:
                        clean[bool_key] = bool(clean[bool_key])
                metadata = ChunkMetadata(**clean)
                self.chunks_list.append(Chunk(text=doc, metadata=metadata))

            self.structure_index = DocumentStructureIndex(self.chunks_list)

            # Build BM25 index
            corpus = [doc.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"Rebuilt BM25 index from {len(documents)} documents in collection")
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")

    
    @staticmethod
    def _reciprocal_rank_fusion(vector_results: List[Tuple[str, float]], bm25_results: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
        """Combine ranked lists using reciprocal rank fusion, matching the legacy retrieval contract."""
        scores: Dict[str, float] = {}
        for ranked_items, weight in ((vector_results, 1.0), (bm25_results, 1.0)):
            for rank, (doc_id, _score) in enumerate(ranked_items, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + (weight / (k + rank))
        return scores

    @staticmethod
    def _format_query(query: str) -> str:
        return f"query: {query}"

    @staticmethod
    def _format_passage(text: str) -> str:
        return f"passage: {text}"

    @staticmethod
    def _meta_dict(metadata) -> dict:
        if isinstance(metadata, dict):
            return metadata
        return metadata.__dict__ if hasattr(metadata, "__dict__") else {}

    @staticmethod
    def _metadata_match_score(meta: dict, analysis: QueryAnalysis) -> float:
        """Boost score when chunk metadata matches section/subsection from the question."""
        boost = 0.0
        if analysis.section_num and meta.get("section_num") == analysis.section_num:
            boost += 0.15
        if analysis.subsection_title and (
            (meta.get("subsection_title") or "").lower()
            == analysis.subsection_title.lower()
        ):
            boost += 0.15
        if analysis.policy_name and meta.get("policy_name") == analysis.policy_name:
            boost += 0.2
        return boost

    def _expand_split_siblings(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """If a split chunk is retrieved, include all parts of the same group."""
        seen = {r["text"] for r in results}
        expanded = list(results)
        group_ids = {
            self._meta_dict(r["metadata"]).get("split_group_id")
            for r in results
        }
        group_ids.discard(None)
        group_ids.discard("")

        for group_id in group_ids:
            for idx, chunk in enumerate(self.chunks_list):
                meta = self._meta_dict(chunk.metadata)
                if meta.get("split_group_id") != group_id:
                    continue
                if chunk.text in seen:
                    continue
                expanded.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": 0.75,
                    "source": self._format_source(chunk.metadata),
                    "_split_sibling": True,
                })
                seen.add(chunk.text)

        def sort_key(r):
            meta = self._meta_dict(r["metadata"])
            return (meta.get("split_part") or 0, -r.get("score", 0))

        expanded.sort(key=sort_key)
        return expanded

    def _structural_candidates(self, analysis: QueryAnalysis) -> List[int]:
        """Return chunks under the exact section/subsection/procedure target."""
        if analysis.intent != "structural":
            return []

        indices: List[int] = []
        for idx, chunk in enumerate(self.chunks_list):
            meta = self._meta_dict(chunk.metadata)

            # Cross-doc scoping: if query was scoped to a doc, skip other docs
            analysis_doc_id = getattr(analysis, "doc_id", None)
            if analysis_doc_id and meta.get("doc_id") and meta["doc_id"] != analysis_doc_id:
                continue

            if analysis.procedure_code:
                if meta.get("procedure_code") != analysis.procedure_code:
                    continue
            else:
                if analysis.section_num is not None:
                    if meta.get("section_num") != analysis.section_num:
                        continue
                if analysis.subsection_num:
                    if meta.get("subsection_num") != analysis.subsection_num:
                        continue
                if analysis.policy_name:
                    if meta.get("policy_name") != analysis.policy_name:
                        continue

            indices.append(idx)

        return indices


    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: float = 0.8,
        bm25_weight: float = 0.2,
        analysis: Optional[QueryAnalysis] = None,
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
        if analysis is None:
            if self.structure_index is None:
                analysis = QueryAnalysis(intent="specific")
            else:
                analysis = analyze_query(query, self.structure_index)

        # Structural queries: exact header match only (no cross-section bleed)
        if analysis.intent == "structural":
            structural_indices = self._structural_candidates(analysis)
            if structural_indices:
                results = []
                for idx in structural_indices:
                    chunk = self.chunks_list[idx]
                    results.append({
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "score": 0.99,
                        "source": self._format_source(chunk.metadata),
                    })
                results = self._expand_split_siblings(results)
                logger.info(
                    "Structural retrieval: section=%s sub=%s proc=%s -> %s chunks",
                    analysis.section_num,
                    analysis.subsection_num,
                    analysis.procedure_code,
                    len(results),
                )
                return results
        total = vector_weight + bm25_weight
        vector_weight /= total
        bm25_weight /= total

        logger.info(
            "Specific retrieval: section=%s sub=%s proc=%s",
            analysis.section_num,
            analysis.subsection_num,
            analysis.procedure_code,
        )
        logger.debug(f"Retrieving top {top_k} chunks for query: {query[:100]}...")

        query_embedding = self.embeddings.encode([self._format_query(query)])[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.retrieval_candidates
        )
        
        vector_scores = {}
        vector_ranked = []
        for doc_id, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
            similarity = 1 - (distance / 2)
            vector_scores[doc_id] = similarity
            vector_ranked.append((doc_id, similarity))
        
        query_tokens = query.lower().split()
        bm25_scores_raw = self.bm25_index.get_scores(query_tokens)
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {self._chunk_id_for(i): score / max_bm25 for i, score in enumerate(bm25_scores_raw)}
        bm25_ranked = [
            (self._chunk_id_for(i), score / max_bm25) for i, score in enumerate(bm25_scores_raw)
        ]
        bm25_ranked = sorted(bm25_ranked, key=lambda item: item[1], reverse=True)[:self.retrieval_candidates]

        combined_scores = {}
        rrf_scores = self._reciprocal_rank_fusion(
            vector_ranked[:self.retrieval_candidates],
            bm25_ranked,
            k=self.rrf_k,
        )
        for doc_id in set(vector_scores) | set(bm25_scores):
            chunk_idx = self._chunk_idx_from_id(doc_id)
            chunk = self.chunks_list[chunk_idx]
            meta = self._meta_dict(chunk.metadata)
            base = (
                vector_weight * vector_scores.get(doc_id, 0)
                + bm25_weight * bm25_scores.get(doc_id, 0)
            )
            boost = self._metadata_match_score(meta, analysis)
            if analysis.intent == "structural":
                boost *= 1.5
            combined_scores[doc_id] = min(1.0, base + boost + rrf_scores.get(doc_id, 0.0))

        # List mode fallback: boost metadata matches inside hybrid results
        if analysis.intent == "structural":
            for idx in self._structural_candidates(analysis):
                doc_id = self._chunk_id_for(idx)
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = 0.85
                else:
                    combined_scores[doc_id] = min(1.0, combined_scores[doc_id] + 0.1)
        
        # Sort and return top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for doc_id, score in sorted_results:
            chunk_idx = self._chunk_idx_from_id(doc_id)
            chunk = self.chunks_list[chunk_idx]
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": score,
                "source": self._format_source(chunk.metadata),
            })


        head = self._expand_split_siblings(results[:5])
        seen = {r["text"] for r in head}
        merged = list(head)
        for r in results[5:]:
            if r["text"] not in seen:
                merged.append(r)
                seen.add(r["text"])
        return merged
    
    @staticmethod
    def _format_source(metadata) -> str:
        meta = HybridRetriever._meta_dict(metadata)
        # Prefer explicit document_title; fall back to doc_id; then a generic label
        doc_title = (
            meta.get("document_title")
            or meta.get("doc_id")
            or "EMPHNET Policy Document"
        )
        parts = [doc_title]

        if meta.get("section_num"):
            parts.append(f"Section {meta['section_num']:02d}")

        if meta.get("policy_name"):
            parts.append(meta["policy_name"])

        if meta.get("procedure_code"):
            parts.append(meta["procedure_code"])

        if meta.get("subsection_num") and meta.get("subsection_title"):
            parts.append(f"{meta['subsection_num']} {meta['subsection_title']}")

        if meta.get("split_part") and meta.get("split_total"):
            parts.append(f"part {meta['split_part']}/{meta['split_total']}")

        if meta.get("page_number"):
            parts.append(f"p. {meta['page_number']}")

        return " • ".join(parts)
    
    def persist(self) -> None:
        """Persist ChromaDB collection to disk"""
        logger.info(f"Persisting collection to {self.chroma_db_path}")
        self.chroma_client.persist()


def load_chunks_from_json(json_path: str) -> List[Chunk]:
    """Load chunks from JSON inspection file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    valid_fields = {f.name for f in ChunkMetadata.__dataclass_fields__.values()}
    for item in data:
        meta = {k: v for k, v in item["metadata"].items() if k in valid_fields}
        metadata = ChunkMetadata(**meta)
        chunks.append(Chunk(text=item["text"], metadata=metadata))
    
    return chunks
