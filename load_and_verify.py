import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())
from src.retrieval import HybridRetriever, load_chunks_from_json

retriever = HybridRetriever(
    chroma_db_path='chroma_storage',
    embedding_model='intfloat/multilingual-e5-large',
    top_k=5
)
retriever.load_or_create_collection('emphnet_policies')

collection_count = retriever.collection.count()
print(f"Collection count: {collection_count}")

if collection_count == 0:
    print("Loading chunks from JSON and embedding...")
    chunks = load_chunks_from_json('data/chunks_inspection.json')
    print(f"Loaded {len(chunks)} chunks")
    retriever.add_chunks(chunks)
    print("Done! Database is ready.")
else:
    print("Collection already has data.")

# Final verification test
print("\n--- VERIFICATION: Sick Leave Query ---")
results = retriever.retrieve("how many days of sick leave per year")
for i, c in enumerate(results):
    score = c.get("score", 0)
    text = c['text'][:300]
    print(f"\nChunk {i+1} (score: {score:.3f}):")
    print(text)
