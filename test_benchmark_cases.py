"""
Test specific benchmark cases to verify exact keyword matching improvements
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import HybridRetriever, load_chunks_from_json
from src.query_intent import analyze_query

# Specific benchmark cases that were failing
BENCHMARK_TEST_CASES = [
    {
        "question": "What software system is used by GHD|EMPHNET for signing electronic documents?",
        "expected_keywords": ["DocuSign", "DSS"],
        "case": "Case 3"
    },
    {
        "question": "What do the acronyms 'KE&NTL' and 'T&ISTL' stand for in the document?",
        "expected_keywords": ["Knowledge Exchange", "Networking Team Leader", "Technology", "Innovative Solutions"],
        "case": "Case 4"
    },
    {
        "question": "What are the 4 main criteria used to evaluate employee performance during appraisals?",
        "expected_keywords": ["Soft Skills", "Job Performance", "SMART Goals", "Job Responsibilities"],
        "case": "Case 1"
    },
    {
        "question": "What are the 6 elements that make up the 'Soft Skills' evaluation criterion?",
        "expected_keywords": ["Reliability", "Accountability", "Civility", "Communication", "Teamwork"],
        "case": "Case 2"
    }
]

def test_benchmark_cases():
    """Test specific benchmark cases with exact keyword matching"""
    
    print("Testing Benchmark Cases for Exact Keyword Matching")
    print("=" * 60)
    
    # Initialize retriever
    print("Initializing retriever...")
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
        print("Vector store is empty. Loading chunks from ingestion...")
        chunks_file = "data/chunks_inspection.json"
        if not Path(chunks_file).exists():
            print(f"Cannot find {chunks_file}. Please run: python src/ingestion.py")
            return
        
        chunks = load_chunks_from_json(chunks_file)
        print(f"Loaded {len(chunks)} chunks from file")
        
        retriever.add_chunks(chunks)
        retriever.persist()
        print("Chunks added to vector store")
    else:
        print(f"Vector store already has {collection_count} chunks")
    
    # Test benchmark cases
    print("\n" + "=" * 60)
    print("Testing Benchmark Cases")
    print("=" * 60)
    
    results_summary = []
    
    for i, test_case in enumerate(BENCHMARK_TEST_CASES, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        case_name = test_case["case"]
        
        print(f"\n{case_name}: {question}")
        print("-" * 60)
        
        # Analyze query
        analysis = analyze_query(question, retriever.structure_index)
        print(f"Query intent: {analysis.intent}")
        
        # Retrieve chunks
        retrieved_chunks = retriever.retrieve(question, analysis=analysis)
        
        # Check if relevant information was found
        found_keywords = []
        for chunk in retrieved_chunks:
            text = chunk["text"].lower()
            for keyword in expected_keywords:
                if keyword.lower() in text:
                    if keyword not in found_keywords:
                        found_keywords.append(keyword)
        
        # Log results
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        print(f"Expected keywords found: {len(found_keywords)}/{len(expected_keywords)}")
        print(f"Found keywords: {found_keywords}")
        
        # Show top chunk
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            print(f"Top chunk score: {top_chunk['score']:.4f}")
            print(f"Top chunk source: {top_chunk['source']}")
            try:
                print(f"Top chunk preview: {top_chunk['text'][:200]}...")
            except UnicodeEncodeError:
                print(f"Top chunk preview: [Text contains special characters]")
        
        # Store results
        results_summary.append({
            "case": case_name,
            "question": question,
            "found_keywords": len(found_keywords),
            "total_keywords": len(expected_keywords),
            "chunks_retrieved": len(retrieved_chunks),
            "top_score": retrieved_chunks[0]['score'] if retrieved_chunks else 0
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for result in results_summary:
        keyword_score = f"{result['found_keywords']}/{result['total_keywords']}"
        print(f"{result['case']}: Keywords={keyword_score}, Chunks={result['chunks_retrieved']}, Score={result['top_score']:.4f}")
    
    # Calculate overall success rate
    keyword_success = sum(1 for r in results_summary if r['found_keywords'] >= r['total_keywords'] * 0.5)
    
    print(f"\nKeyword match success rate: {keyword_success}/{len(results_summary)} ({keyword_success/len(results_summary)*100:.1f}%)")
    
    print("\nBenchmark testing complete!")

if __name__ == "__main__":
    test_benchmark_cases()