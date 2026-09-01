"""
Test the WFA question specifically to ensure it doesn't return leave policies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import HybridRetriever, load_chunks_from_json
from src.query_intent import analyze_query

# Test the specific WFA question
WFA_QUESTION = "If an employee asks to work remotely for a week, under what policy section can I approve this, through which system must it be submitted, and what type of leave does it count as?"

def test_wfa_question():
    """Test that WFA question returns correct policy section"""
    
    print("Testing WFA vs Leave Policy Confusion")
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
    
    # Test WFA question
    print(f"\nQuestion: {WFA_QUESTION}")
    print("-" * 60)
    
    # Analyze query
    analysis = analyze_query(WFA_QUESTION, retriever.structure_index)
    print(f"Query intent: {analysis.intent}")
    print(f"Matched section: {analysis.section_title}")
    print(f"Matched subsection: {analysis.subsection_title}")
    print(f"Matched policy: {analysis.policy_name}")
    print(f"Analysis details: {analysis}")
    
    # Retrieve chunks
    retrieved_chunks = retriever.retrieve(WFA_QUESTION, analysis=analysis)
    
    # Check for leave policy contamination
    leave_keywords = ["personal leave", "annual leave", "sick leave", "vacation", "hourly leave"]
    wfa_keywords = ["working from anywhere", "wfa", "remote work", "flexible work"]
    
    print(f"\nRetrieved {len(retrieved_chunks)} chunks")
    
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
    
    # Check for leave vs WFA content
    has_leave_content = False
    has_wfa_content = False
    
    for chunk in retrieved_chunks:
        text = chunk["text"].lower()
        for keyword in leave_keywords:
            if keyword in text:
                has_leave_content = True
                break
        for keyword in wfa_keywords:
            if keyword in text:
                has_wfa_content = True
                break
    
    print(f"Contains leave policy content: {has_leave_content}")
    print(f"Contains WFA content: {has_wfa_content}")
    
    # Show top chunk
    if retrieved_chunks:
        top_chunk = retrieved_chunks[0]
        print(f"\nTop chunk score: {top_chunk['score']:.4f}")
        print(f"Top chunk source: {top_chunk['source']}")
        try:
            print(f"Top chunk preview: {top_chunk['text'][:300]}...")
        except UnicodeEncodeError:
            print(f"Top chunk preview: [Text contains special characters]")
    
    # Final assessment
    if has_wfa_content and not has_leave_content:
        print("\nWFA Test: PASS - Returns WFA policy without leave contamination")
    elif has_wfa_content and has_leave_content:
        print("\nWFA Test: PARTIAL - Returns WFA but with some leave content")
    else:
        print("\nWFA Test: FAIL - Does not return WFA policy correctly")

if __name__ == "__main__":
    test_wfa_question()