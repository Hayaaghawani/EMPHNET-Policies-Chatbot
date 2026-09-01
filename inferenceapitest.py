import os

from huggingface_hub import InferenceClient

print("1. Starting...")

api_key = os.getenv("HF_API_KEY")
if not api_key:
    raise RuntimeError("Set HF_API_KEY in your environment before running this probe.")

client = InferenceClient(api_key=api_key, provider="auto")

print("2. Client created")
print("3. Sending request...")

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {
            "role": "user",
            "content": "Say hello and tell me what model you are."
        }
    ],
)

print("4. Response received")
print(response)
print("5. Answer:")
print(response.choices[0].message.content)
