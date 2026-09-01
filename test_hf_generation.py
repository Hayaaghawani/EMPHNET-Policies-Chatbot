"""
Test HF API generation with actual prompts to debug the issue
"""

import os

from huggingface_hub import InferenceClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("HF_API_KEY")
if not api_key:
    raise RuntimeError("Set HF_API_KEY in your environment before running this manual probe.")

client = InferenceClient(api_key=api_key, provider="auto")

# Test with actual system prompt that fails
system_prompt = """You are an EMPHNET HR policy assistant. Answer ONLY from the retrieved text.

CRITICAL RULES:
1. Give a focused answer with the specific details asked (numbers, conditions, steps).
2. Never say "not found" if the answer is in the context.
3. For how-to questions, list only the relevant steps in order.
4. Do NOT invent information or use external knowledge.
5. Do NOT make speculative deductions (e.g., "implied", "likely", "probably").
6. If a specific metric, scale, or step is missing from the retrieved context, explicitly state that it is not in the context.
7. Never use standard HR terminology or assumptions to fill gaps in the retrieved text.

Format: direct answer, brief quote, source reference."""

user_prompt = """Question: What are the procedures for hiring a full time employee?

Context:
Source 1: HR Policies & Procedures Manual – Section 1 – Subsection 1.5 – (ML-HR-01.P03) – p. 28
[Section 01: RECRUITMENT & SELECTION | 1.5 Procedures | Procedure 03: Hire Full Time Employees | ML-HR-01.P03]
Procedure
Reference
Code
ML-ML-01.P03 Hire Full Time
Employees
Manual
Version no.:
01
1. When a vacancy is announced, HR Team to post the vacancy announcement on one or more of the following platforms:
- Official website of GHD|EMPHNET
- Social media channels
- Professional networks
- Partner organizations

Answer the question using ONLY the context above. Give a focused answer."""

print("Testing HF API with actual prompts...")
print("=" * 60)

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=400,
    )
    
    print("Response type:", type(response))
    print("Response:", response)
    
    # Handle different response formats
    if hasattr(response, 'choices') and response.choices:
        answer = response.choices[0].message.content
        print("Answer from choices:", answer)
    elif isinstance(response, dict):
        answer = response.get('content', str(response))
        print("Answer from dict:", answer)
    else:
        answer = str(response)
        print("Answer from str:", answer)
    
    print("HF API test successful!")
    
except Exception as e:
    print(f"HF API test failed: {e}")
    import traceback
    traceback.print_exc()
