"""
Test structural questions to verify the original functionality is restored
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import HybridRetriever, load_chunks_from_json
from src.query_intent import analyze_query

# Key structural questions that were working before
STRUCTURAL_TEST_QUESTIONS = [
    {
        "question": "What are the responsibilities of HR regarding recruitment and selection?",
        "expected_section": "RECRUITMENT & SELECTION",
        "expected_subsection": "Responsibilities"
    },
    {
        "question": "What are the procedures for hiring a full time employee?",
        "expected_procedure": "Hire Full Time Employees",
        "expected_section": "RECRUITMENT & SELECTION"
    },
    {
        "question": "What are the clearance steps for an employee leaving the company?",
        "expected_procedure": "End of Contract",
        "expected_section": "EMPLOYMENT CONTRACT"
    },
    {
        "question": "What are the responsibilities under the attendance and leaves section?",
        "expected_section": "ATTENDANCE, LEAVES & VACATIONS",
        "expected_subsection": "Responsibilities"
    }
]

def test_structural_awareness():
    """Test that structural awareness is restored"""
    
    print("Testing Structural Awareness Restoration")
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
    
    # Test structural questions
    print("\n" + "=" * 60)
    print("Testing Structural Questions")
    print("=" * 60)
    
    for i, test_case in enumerate(STRUCTURAL_TEST_QUESTIONS, 1):
        question = test_case["question"]
        expected_section = test_case.get("expected_section")
        expected_subsection = test_case.get("expected_subsection")
        expected_procedure = test_case.get("expected_procedure")
        
        print(f"\nTest {i}: {question}")
        print("-" * 60)
        
        # Analyze query
        analysis = analyze_query(question, retriever.structure_index)
        print(f"Query intent: {analysis.intent}")
        print(f"Matched section: {analysis.section_title}")
        print(f"Matched subsection: {analysis.subsection_title}")
        print(f"Matched procedure: {analysis.procedure_name}")
        
        # Retrieve chunks
        retrieved_chunks = retriever.retrieve(question, analysis=analysis)
        
        # Check section consistency
        section_consistent = True
        sections_found = set()
        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            if hasattr(metadata, 'section_title'):
                if metadata.section_title:
                    sections_found.add(metadata.section_title)
            elif isinstance(metadata, dict):
                if metadata.get('section_title'):
                    sections_found.add(metadata.get('section_title'))
        
        print(f"Sections retrieved: {sections_found}")
        
        if expected_section and expected_section not in str(sections_found):
            print(f"WARNING: Expected section {expected_section} not found in retrieved sections")
            section_consistent = False
        
        # Check for cross-contamination
        if len(sections_found) > 1:
            print(f"WARNING: Multiple sections retrieved - potential cross-contamination")
            section_consistent = False
        
        # Show top chunk
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            print(f"Top chunk score: {top_chunk['score']:.4f}")
            print(f"Top chunk source: {top_chunk['source']}")
            print(f"Top chunk preview: {top_chunk['text'][:200]}...")
        
        status = "PASS" if section_consistent else "FAIL"
        print(f"Structural consistency: {status}")
    
    print("\n" + "=" * 60)
    print("Structural awareness testing complete!")

if __name__ == "__main__":
    test_structural_awareness()