"""Quick retrieval verification after pipeline improvements."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.retrieval import HybridRetriever

retriever = HybridRetriever(top_k=8)
retriever.load_or_create_collection('emphnet_policies')

queries = [
    "how many days hajj leave",
    "what are the steps to request time off leave",
    "what is the sick leave policy",
]

for q in queries:
    print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
    results = retriever.retrieve(q)
    for i, c in enumerate(results[:3], 1):
        print(f"\n[{i}] score={c['score']:.3f} | {c['source']}")
        print(c['text'][:500])
