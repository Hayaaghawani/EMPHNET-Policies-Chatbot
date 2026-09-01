"""
Test that hiring procedures return all chunks for Procedure 3, not mixed with other procedures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import HybridRetriever, load_chunks_from_json
from src.query_intent import analyze_query

HIRING_QUESTION = "What are the procedures for hiring a full time employee?"

def test_hiring_procedure():
    """Test that hiring procedures return only Procedure 3 chunks, not other procedures"""
    
    print("Testing Hiring Procedure Retrieval")
    print("=" * 60)
    
    # Initialize retriever
    print("Initializing retriever...")
    retriever = HybridRetriever(
        chroma_db_path="chroma_storage",
        embedding_model="intfloat/multilingual-e5-large",
        top_k=20,  # Get more chunks to see if we get all procedure parts
        rrf_k=60,
        retrieval_candidates=50  # Get more candidates to find all procedure parts
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
    
    # Test hiring question
    print(f"\nQuestion: {HIRING_QUESTION}")
    print("-" * 60)
    
    # Analyze query
    analysis = analyze_query(HIRING_QUESTION, retriever.structure_index)
    print(f"Query intent: {analysis.intent}")
    print(f"Matched section: {analysis.section_title}")
    print(f"Matched subsection: {analysis.subsection_title}")
    print(f"Matched procedure: {analysis.procedure_name}")
    print(f"Procedure code: {analysis.procedure_code}")
    
    # Retrieve chunks
    retrieved_chunks = retriever.retrieve(HIRING_QUESTION, analysis=analysis)
    
    print(f"\nRetrieved {len(retrieved_chunks)} chunks")
    
    # Check for procedure contamination
    other_procedures = set()
    procedure_codes = set()
    
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        if hasattr(metadata, 'procedure_code'):
            if metadata.procedure_code:
                procedure_codes.add(metadata.procedure_code)
                if metadata.procedure_code != "ML-HR-01.P03":
                    other_procedures.add(metadata.procedure_code)
        elif isinstance(metadata, dict):
            if metadata.get('procedure_code'):
                procedure_codes.add(metadata.get('procedure_code'))
                if metadata.get('procedure_code') != "ML-HR-01.P03":
                    other_procedures.add(metadata.get('procedure_code'))
    
    print(f"Procedure codes found: {procedure_codes}")
    print(f"Other procedures found: {other_procedures}")
    
    # Check for split group IDs
    split_groups = set()
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        if hasattr(metadata, 'split_group_id'):
            if metadata.split_group_id:
                split_groups.add(metadata.split_group_id)
        elif isinstance(metadata, dict):
            if metadata.get('split_group_id'):
                split_groups.add(metadata.get('split_group_id'))
    
    print(f"Split groups found: {split_groups}")
    
    # Show chunk details
    print(f"\nChunk details:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk["metadata"]
        if hasattr(metadata, 'procedure_code'):
            proc_code = metadata.procedure_code
            split_group = metadata.split_group_id
            split_part = metadata.split_part
        else:
            proc_code = metadata.get('procedure_code')
            split_group = metadata.get('split_group_id')
            split_part = metadata.get('split_part')
        
        print(f"  Chunk {i}: Procedure={proc_code}, SplitGroup={split_group}, Part={split_part}")
        print(f"    Source: {chunk['source']}")
        try:
            print(f"    Preview: {chunk['text'][:100]}...")
        except UnicodeEncodeError:
            print(f"    Preview: [Text contains special characters]")
    
    # Final assessment
    if not other_procedures and len(procedure_codes) == 1 and "ML-HR-01.P03" in procedure_codes:
        print("\nHiring Procedure Test: PASS - Only Procedure 3 chunks retrieved")
    elif other_procedures:
        print(f"\nHiring Procedure Test: FAIL - Retrieved chunks from other procedures: {other_procedures}")
    else:
        print("\nHiring Procedure Test: PARTIAL - Results unclear")

if __name__ == "__main__":
    test_hiring_procedure()