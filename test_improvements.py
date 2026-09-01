"""
Test script to verify RAG pipeline improvements based on benchmark failures
"""

import sys
import logging
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import HybridRetriever, load_chunks_from_json
from src.generation import OllamaGenerator
from src.query_intent import analyze_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Benchmark test cases from the document
BENCHMARK_QUESTIONS = [
    {
        "question": "What are the 4 main criteria used to evaluate employee performance during appraisals?",
        "expected_keywords": ["Soft Skills", "Job Performance", "SMART Goals", "Job Responsibilities", "Adherence to Organizational Policies"],
        "expected_section": "Section 02"
    },
    {
        "question": "What are the 6 elements that make up the 'Soft Skills' evaluation criterion?",
        "expected_keywords": ["Reliability", "Accountability", "Civility", "Communication", "Teamwork"],
        "expected_section": "Section 02"
    },
    {
        "question": "What software system is used by GHD|EMPHNET for signing electronic documents?",
        "expected_keywords": ["DocuSign", "DSS"],
        "expected_section": "Section 03"
    },
    {
        "question": "What do the acronyms 'KE&NTL' and 'T&ISTL' stand for in the document?",
        "expected_keywords": ["Knowledge Exchange", "Networking Team Leader", "Technology", "Innovative Solutions"],
        "expected_section": "Definitions"
    },
    {
        "question": "Under Section 07 (Personnel Management), what three standard procedures are detailed?",
        "expected_keywords": ["Health Insurance", "Experience", "Employment", "Salary", "Recommendation Letter"],
        "expected_section": "Section 07"
    },
    {
        "question": "How long is the standard probation period for newly recruited employees?",
        "expected_keywords": ["3 months"],
        "expected_section": "Section 01"
    }
]

def test_retrieval_improvements():
    """Test the improved retrieval system"""
    
    logger.info("Testing RAG Pipeline Improvements")
    logger.info("=" * 60)
    
    # Initialize retriever
    logger.info("Initializing retriever...")
    retriever = HybridRetriever(
        chroma_db_path="chroma_storage",
        embedding_model="intfloat/multilingual-e5-large",
        top_k=5,
        rrf_k=60,
        retrieval_candidates=20
    )
    
    # Load or create collection
    retriever.load_or_create_collection("emphnet_policies")
    
    # Check if collection has data
    collection_count = retriever.collection.count()
    if collection_count == 0:
        logger.info("Vector store is empty. Loading chunks from ingestion...")
        chunks_file = "data/chunks_inspection.json"
        if not Path(chunks_file).exists():
            logger.error(f"Cannot find {chunks_file}. Please run: python src/ingestion.py")
            return
        
        chunks = load_chunks_from_json(chunks_file)
        logger.info(f"Loaded {len(chunks)} chunks from file")
        
        retriever.add_chunks(chunks)
        retriever.persist()
        logger.info("Chunks added to vector store")
    else:
        logger.info(f"Vector store already has {collection_count} chunks")
    
    # Enable reranking
    try:
        retriever.enable_reranker()
        logger.info("Reranking enabled")
    except Exception as e:
        logger.warning(f"Reranking not available: {e}")
    
    # Test benchmark questions
    logger.info("\n" + "=" * 60)
    logger.info("Testing Benchmark Questions")
    logger.info("=" * 60)
    
    results_summary = []
    
    for i, test_case in enumerate(BENCHMARK_QUESTIONS, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_section = test_case["expected_section"]
        
        logger.info(f"\nTest {i}: {question}")
        logger.info("-" * 60)
        
        # Analyze query
        analysis = analyze_query(question, retriever.structure_index)
        logger.info(f"Query intent: {analysis.intent}")
        
        # Retrieve chunks
        retrieved_chunks = retriever.retrieve(question, analysis=analysis, use_parent_context=True)
        
        # Check if relevant information was found
        found_keywords = []
        for chunk in retrieved_chunks:
            text = chunk["text"].lower()
            for keyword in expected_keywords:
                if keyword.lower() in text:
                    if keyword not in found_keywords:
                        found_keywords.append(keyword)
        
        # Check if expected section was retrieved
        found_section = False
        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            if hasattr(metadata, 'section_title'):
                if expected_section.lower() in str(metadata.section_title).lower():
                    found_section = True
                    break
            elif isinstance(metadata, dict):
                if expected_section.lower() in str(metadata.get('section_title', '')).lower():
                    found_section = True
                    break
        
        # Log results
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        logger.info(f"Expected keywords found: {len(found_keywords)}/{len(expected_keywords)}")
        logger.info(f"Found keywords: {found_keywords}")
        logger.info(f"Expected section found: {found_section}")
        
        # Show top chunk
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            logger.info(f"Top chunk score: {top_chunk['score']:.4f}")
            logger.info(f"Top chunk source: {top_chunk['source']}")
            logger.info(f"Top chunk preview: {top_chunk['text'][:200]}...")
        
        # Store results
        results_summary.append({
            "question": question,
            "found_keywords": len(found_keywords),
            "total_keywords": len(expected_keywords),
            "found_section": found_section,
            "chunks_retrieved": len(retrieved_chunks),
            "top_score": retrieved_chunks[0]['score'] if retrieved_chunks else 0
        })
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for i, result in enumerate(results_summary, 1):
        keyword_score = f"{result['found_keywords']}/{result['total_keywords']}"
        section_status = "✓" if result['found_section'] else "✗"
        logger.info(f"Test {i}: Keywords={keyword_score}, Section={section_status}, Chunks={result['chunks_retrieved']}, Score={result['top_score']:.4f}")
    
    # Calculate overall success rate
    keyword_success = sum(1 for r in results_summary if r['found_keywords'] >= r['total_keywords'] * 0.5)
    section_success = sum(1 for r in results_summary if r['found_section'])
    
    logger.info(f"\nKeyword match success rate: {keyword_success}/{len(results_summary)} ({keyword_success/len(results_summary)*100:.1f}%)")
    logger.info(f"Section match success rate: {section_success}/{len(results_summary)} ({section_success/len(results_summary)*100:.1f}%)")
    
    logger.info("\n✓ Testing complete!")

if __name__ == "__main__":
    test_retrieval_improvements()