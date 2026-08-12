import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv

load_dotenv()

from src.retrieval import HybridRetriever
from src.generation import OllamaGenerator

q = "how many days hajj leave"
r = HybridRetriever(top_k=8)
r.load_or_create_collection("emphnet_policies")
chunks = r.retrieve(q)

gen = OllamaGenerator()
t0 = time.time()
result = gen.generate(q, chunks)
print(f"Time: {time.time() - t0:.1f}s")
print(result["answer"][:400])
