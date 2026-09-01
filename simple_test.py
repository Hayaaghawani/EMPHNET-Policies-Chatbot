"""
Simple test to verify the improved ingestion and chunking
"""

import json
from pathlib import Path

# Load the newly created chunks
chunks_file = "data/chunks_inspection.json"
with open(chunks_file, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total chunks created: {len(chunks)}")

# Check for new metadata fields
parent_chunks = [c for c in chunks if c['metadata'].get('chunk_level') == 'parent']
child_chunks = [c for c in chunks if c['metadata'].get('chunk_level') == 'child']

print(f"Parent chunks: {len(parent_chunks)}")
print(f"Child chunks: {len(child_chunks)}")

# Check for acronyms and keywords
chunks_with_acronyms = [c for c in chunks if c['metadata'].get('acronyms')]
chunks_with_keywords = [c for c in chunks if c['metadata'].get('keywords')]

print(f"Chunks with acronyms: {len(chunks_with_acronyms)}")
print(f"Chunks with keywords: {len(chunks_with_keywords)}")

# Show examples
if chunks_with_acronyms:
    print("\nExample chunk with acronyms:")
    example = chunks_with_acronyms[0]
    print(f"Acronyms: {example['metadata']['acronyms']}")
    print(f"Text preview: {example['text'][:200]}...")

if chunks_with_keywords:
    print("\nExample chunk with keywords:")
    example = chunks_with_keywords[0]
    print(f"Keywords: {example['metadata']['keywords']}")
    print(f"Text preview: {example['text'][:200]}...")

# Look for specific test cases
print("\nSearching for DocuSign/DSS...")
dss_chunks = [c for c in chunks if 'dss' in c['text'].lower() or 'docusign' in c['text'].lower()]
print(f"Found {len(dss_chunks)} chunks mentioning DocuSign/DSS")
if dss_chunks:
    print(f"Example: {dss_chunks[0]['text'][:300]}...")

print("\nSearching for KE&NTL...")
kentl_chunks = [c for c in chunks if 'ke&ntl' in c['text'].lower()]
print(f"Found {len(kentl_chunks)} chunks mentioning KE&NTL")
if kentl_chunks:
    print(f"Example: {kentl_chunks[0]['text'][:300]}...")

print("\nSearching for Soft Skills...")
soft_skills_chunks = [c for c in chunks if 'soft skills' in c['text'].lower()]
print(f"Found {len(soft_skills_chunks)} chunks mentioning Soft Skills")
if soft_skills_chunks:
    # Safe printing for unicode
    try:
        print(f"Example: {soft_skills_chunks[0]['text'][:300]}...")
    except UnicodeEncodeError:
        print(f"Example: [Text contains special characters]")

print("\nIngestion verification complete!")