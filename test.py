import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())
from src.retrieval import HybridRetriever, load_chunks_from_json

retriever = HybridRetriever(chroma_db_path='chroma_storage', embedding_model='intfloat/multilingual-e5-large', top_k=3)
retriever.load_or_create_collection('emphnet_policies')

collection_count = retriever.collection.count()
if collection_count == 0:
    chunks = load_chunks_from_json('data/chunks_inspection.json')
    retriever.add_chunks(chunks)
    retriever.persist()

chunks = retriever.retrieve('What is the sick leave policy?')
print(f'\nTotal retrieved: {len(chunks)}')
for i, c in enumerate(chunks):
    print(f'\n--- Chunk {i+1} ---')
    print(f'Score: {c.get("score")}')
    print(c['text'][:400])
