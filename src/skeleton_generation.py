"""LLM navigation and grounded answer generation for skeleton nodes."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from .skeleton_pipeline_types import Navigation


class SkeletonLLM:
    """Provider adapter used for outline navigation and grounded generation."""

    def __init__(self, *, hf_api_key: str = "", hf_model: str = "Qwen/Qwen3-32B",
                 ollama_host: str = "http://localhost:11434", ollama_model: str = "llama3.1:latest",
                 timeout: int = 300):
        self.hf_api_key = hf_api_key.strip().strip('"').strip("'")
        self.hf_model = hf_model
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.timeout = timeout

    def _complete(self, system: str, user: str) -> str:
        if self.hf_api_key:
            response = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.hf_api_key}"},
                json={"model": self.hf_model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.0, "max_tokens": 1200},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json={"model": self.ollama_model, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "options": {"temperature": 0}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def navigate(
        self,
        question: str,
        outline: list[dict[str, str]],
        similarity_candidates: list[dict[str, Any]] | None = None,
    ) -> Navigation:
        candidates = similarity_candidates or []
        raw = self._complete(
            "Return only valid JSON. Select the smallest relevant node set. broad=true means the full node/subtree is needed; broad=false means one node's own_text is enough. Use semantic candidates as evidence, but choose any outline node when the candidates miss the question.",
            f"Question: {question}\nOutline:\n{json.dumps(outline, ensure_ascii=False)}\nSemantic candidates:\n{json.dumps(candidates, ensure_ascii=False)}\nReturn {{\"node_ids\":[{{\"doc_id\":\"...\",\"id\":\"...\"}}],\"broad\":false}}",
        )
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError(f"Navigator did not return JSON: {raw[:200]}")
        payload = json.loads(match.group(0))
        refs = [(item["doc_id"], item["id"]) for item in payload.get("node_ids", [])]
        known = {(item["doc_id"], item["id"]) for item in outline}
        refs = [ref for ref in refs if ref in known]
        if not refs:
            raise ValueError("Navigator returned no known node IDs")
        return Navigation(refs, bool(payload.get("broad")))

    def answer(self, question: str, contexts: list[dict[str, str]]) -> dict[str, Any]:
        context = "\n\n---\n\n".join(f"PATH: {item['path']}\n{item['text']}" for item in contexts)
        raw = self._complete(
            "Answer only from the supplied text. If it does not answer the question, say that explicitly. Answer in the same language as the question. Do not add citations or facts not present in the text.",
            f"Question: {question}\n\nSUPPLIED TEXT:\n{context}",
        )
        return {"answer": raw.strip(), "sources": contexts}
