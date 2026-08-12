import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, json
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())
from src.retrieval import HybridRetriever

retriever = HybridRetriever(chroma_db_path='chroma_storage', embedding_model='intfloat/multilingual-e5-large', top_k=8)
retriever.load_or_create_collection('emphnet_policies')

queries = [
    "how many days hajj leave",
    "steps to request time off leave",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    results = retriever.retrieve(q)
    for i, c in enumerate(results):
        print(f"\n[Chunk {i+1}] score={c.get('score',0):.3f}")
        print(c['text'][:400])

# Also show a sample of 5 random chunks to see their quality
print("\n\n" + "="*60)
print("SAMPLE OF 5 CHUNKS FROM DATABASE (to check quality):")
print("="*60)
with open('data/chunks_inspection.json', encoding='utf-8') as f:
    all_chunks = json.load(f)

# Find Hajj chunk
for i, c in enumerate(all_chunks):
    if 'hajj' in c['text'].lower():
        print(f"\n--- HAJJ CHUNK (index {i}) ---")
        print(f"Text: {c['text']}")
        print(f"Metadata section_num: {c['metadata'].get('section_num')}")
        print(f"Metadata section_title: {c['metadata'].get('section_title')}")
        break
