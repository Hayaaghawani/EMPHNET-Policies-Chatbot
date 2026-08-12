import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())
from src.retrieval import HybridRetriever

retriever = HybridRetriever(chroma_db_path='chroma_storage', embedding_model='intfloat/multilingual-e5-large', top_k=15)
retriever.load_or_create_collection('emphnet_policies')
chunks = retriever.retrieve('duration of sick leave how many days')
found = False
for i, c in enumerate(chunks):
    if '14 days' in c['text'].lower():
        print(f'FOUND IN CHUNK {i+1} with score {c.get("score")}')
        print(c['text'])
        found = True

if not found:
    print('NOT FOUND IN TOP 15')
