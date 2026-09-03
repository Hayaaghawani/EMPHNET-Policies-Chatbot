"""
Generation Module - LLM-based Answer Generation with Grounding

Generates answers strictly from retrieved policy chunks.
Prioritizes Hugging Face Serverless Router; falls back to local Ollama.
"""

import logging
import os
import re
import requests
from typing import List, Dict, Any, Tuple, Optional

from .query_intent import QueryAnalysis, analyze_query

logger = logging.getLogger(__name__)


class ProviderFailure(RuntimeError):
    """Raised when a generation provider fails."""


class BaseGenerator:
    """Shared utilities, prompts, formatting, and heuristic scoring."""

    @staticmethod
    def _prepare_chunks_for_llm(
        retrieved_chunks: List[Dict[str, Any]],
        analysis: QueryAnalysis,
        max_chunks: int,
        max_chars_per_chunk: int,
    ) -> List[Dict[str, Any]]:
        if analysis.intent in ("list", "structural"):
            limit = max_chunks
            char_limit = max_chars_per_chunk
        else:
            limit = min(max_chunks, 5)
            char_limit = max_chars_per_chunk

        prepared = []
        for chunk in retrieved_chunks[:limit]:
            trimmed = dict(chunk)
            text = chunk.get("text", "")
            if analysis.intent not in ("list", "structural") and len(text) > char_limit:
                trimmed["text"] = text[:char_limit] + "\n[...truncated...]"
            prepared.append(trimmed)
        return prepared


    @staticmethod
    def _detect_language(text: str) -> str:
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_count > len(text) * 0.3 else "en"

    @staticmethod
    def _build_system_prompt(language: str, analysis: QueryAnalysis) -> str:
        if language == "ar":
            return """أنت مساعد متخصص في سياسات الموارد البشرية لمنظمة EMPHNET. أجب بناءً على النصوص المُسترجعة فقط.

قواعد صارمة:
1. اقرأ جميع النصوص المُسترجعة قبل الإجابة.
2. إذا وُجدت الإجابة في أي نص مُسترجع، قدّم كل التفاصيل (أرقام، شروط، خطوات). ممنوع قول "لا أعرف" إذا المعلومة موجودة.
3. اقتبس الأرقام والمدد حرفياً (مثلاً: "14 يوماً").
4. للأسئلة الإجرائية والقوائم: اذكر الخطوات المرقمة بالترتيب من جميع النصوص ذات الصلة.
5. لا تختلق معلومات غير موجودة في النصوص.
6. الرد بالعربية.

قواعد التنسيق الإجبارية:
- يجب وضع كل خطوة أو نقطة في سطر مستقل تماماً.
- افصل بين كل نقطة مرقمة والأخرى بسطر فارغ (Double Line Break).
- ممنوع دمج النقاط في فقرة واحدة متصلة.
- في النهاية اذكر المرجع والمصدر."""
        else:
            if analysis.intent in ("list", "structural"):
                return """You are an EMPHNET HR policy assistant. The user wants a COMPLETE LIST of items from the retrieved text.

Rules:
1. List ALL items, points, or bullet points (numbered 1,2,3 or bulleted ➢,-,•) found in the context — do NOT skip items.
2. Present each item clearly on its own line.
3. Stay strictly within the retrieved context — do not invent items.
4. Merge related chunks (e.g. Part 1 and Part 2) into one complete list.

MANDATORY FORMATTING:
- Put EACH item or step on its OWN line.
- Leave an empty line between items (use double newlines).
- NEVER repeat or loop text.
- End with the source reference."""

            return """You are an EMPHNET HR policy assistant. Answer ONLY from the retrieved text.

Rules:
1. Answer the EXACT question asked — only the specific fact, number, deadline, or rule requested.
2. Do NOT list all retrieved content. Extract only the relevant detail.
3. If multiple chunks are retrieved, read all of them but quote only what directly answers the question.
4. For multi-step or procedural questions, put each step on its own line.
5. Never invent information. Never say "not found" if the answer exists in context.

Format: one focused answer; cite the source section at the end."""

    @staticmethod
    def _build_user_prompt(query: str, context: str, language: str, analysis: QueryAnalysis) -> str:
        if language == "ar":
            return f"""السؤال: {query}

السياق:
{context}

الرجاء الإجابة على السؤال بناءً على السياق المقدم أعلاه فقط مع فصل كل نقطة مرقمة في سطر مستقل."""
        if analysis.intent in ("list", "structural"):
            return f"""Question: {query}

Context:
{context}

List EVERY numbered point from the context above. 
CRITICAL: Put each numbered point on a NEW line. Do NOT combine them into one paragraph."""
        return f"""Question: {query}

Context:
{context}

Answer the question using ONLY the context above. Use clear line breaks between distinct steps or points."""

    @staticmethod
    def _format_context(retrieved_chunks: List[Dict[str, Any]], language: str) -> str:
        if language == "ar":
            header = "المستندات ذات الصلة:\n"
        else:
            header = "Retrieved Policy Documents:\n"
        separator = "\n---\n"

        context_parts = [header]
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get('source', 'Unknown Source')
            text = chunk.get('text', '')
            if language == "ar":
                context_parts.append(f"({i}) مصدر: {source}\nالنص:\n{text}")
            else:
                context_parts.append(f"({i}) Source: {source}\nText:\n{text}")
        return separator.join(context_parts)

    @staticmethod
    def _parse_response(response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        parts = response_text.split("[Source:")
        answer = parts[0].strip()

        # Deterministic line-break fix:
        # If the LLM collapses numbered items (e.g., "...etc. 2. If ... 3. Prepare ..."),
        # automatically inject clean double newlines before digits followed by dots or bullets.
        answer = re.sub(r'([^\n])\s+(\d{1,2}\.\s+[A-Z\u0621-\u064A])', r'\1\n\n\2', answer)
        answer = re.sub(r'([^\n])\s+(•\s+)', r'\1\n\n\2', answer)

        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata')
            if not isinstance(metadata, dict):
                metadata = {}
            source_text = chunk.get('source', 'Unknown')
            text_excerpt = chunk.get('text', '')[:300] + "..."
            sources.append({
                "section": metadata.get('section_num', 'Unknown'),
                "source": source_text,
                "excerpt": text_excerpt,
            })
        return answer, sources

    @staticmethod
    def _calculate_confidence(response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        confidence = 0.7
        refusal_phrases = ["couldn't find", "not found", "not mentioned", "no information", "unable to find"]
        if any(phrase in response_text.lower() for phrase in refusal_phrases):
            confidence -= 0.3

        num_chunks = len(retrieved_chunks)
        if num_chunks < 2:
            confidence -= 0.2
        elif num_chunks >= 5:
            confidence += 0.1

        if retrieved_chunks:
            avg_score = sum(c.get('score', 0) for c in retrieved_chunks) / len(retrieved_chunks)
            confidence += (avg_score - 0.5) * 0.2

        return max(0.0, min(1.0, confidence))


class OllamaGenerator(BaseGenerator):
    """Local Ollama generation provider."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.1:latest",
        temperature: float = 0.3,
        top_p: float = 0.9,
        timeout: int = 300,
        max_chunks: int = 5,
        max_chars_per_chunk: int = 3500,
        num_predict: int = 1800,
        list_max_chunks: int = 8,
        list_max_chars_per_chunk: int = 6000,
        list_num_predict: int = 2500,
        **kwargs: Any,
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.num_predict = num_predict
        self.list_max_chunks = list_max_chunks
        self.list_max_chars_per_chunk = list_max_chars_per_chunk
        self.list_num_predict = list_num_predict

        self._soft_verify_and_warmup()

    def _soft_verify_and_warmup(self) -> None:
        """Non-blocking check so Ollama outage doesn't block startup if HF is primary."""
        try:
            res = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            if res.status_code == 200:
                logger.info(f"Connected to Ollama at {self.ollama_host}")
                requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.model, "prompt": "ok", "stream": False, "options": {"num_predict": 1}},
                    timeout=5,
                )
        except Exception as e:
            logger.warning(f"Ollama local service not immediately reachable ({e}). Will retry on demand.")

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        analysis: Optional[QueryAnalysis] = None,
    ) -> Dict[str, Any]:
        analysis = analysis or analyze_query(query)
        language = self._detect_language(query)

        if analysis.intent in ("list", "structural"):
            max_chunks, max_chars, num_predict = self.list_max_chunks, self.list_max_chars_per_chunk, self.list_num_predict
        else:
            max_chunks, max_chars, num_predict = self.max_chunks, self.max_chars_per_chunk, self.num_predict

        llm_chunks = self._prepare_chunks_for_llm(retrieved_chunks, analysis, max_chunks, max_chars)
        context = self._format_context(llm_chunks, language)
        system_prompt = self._build_system_prompt(language, analysis)
        user_prompt = self._build_user_prompt(query, context, language, analysis)

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "keep_alive": "10m",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "options": {
                "num_predict": num_predict,
                "num_ctx": 8192 if num_predict > 500 else 4096,
                "repeat_penalty": 1.15,
            },

        }

        try:
            res = requests.post(url, json=payload, timeout=self.timeout)
            res.raise_for_status()
            response_text = res.json().get("response", "")
        except Exception as e:
            raise ProviderFailure(f"Ollama error: {e}")

        answer, sources = self._parse_response(response_text, retrieved_chunks)
        confidence = self._calculate_confidence(response_text, retrieved_chunks)

        return {
            "answer": answer,
            "language": language,
            "sources": sources,
            "confidence": confidence,
            "provider": "ollama",
            "raw_response": response_text,
            "metadata": {},
        }


class HuggingFaceProvider(BaseGenerator):
    """Hugging Face Inference Router generation provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen3-32B",
        provider: str = "nscale",
        max_chunks: int = 5,
        max_chars_per_chunk: int = 3500,
        list_max_chunks: int = 8,
        list_max_chars_per_chunk: int = 6000,
        temperature: float = 0.3,
        timeout: int = 120,
        **kwargs: Any,
    ):
        self.api_key = api_key.strip().strip('"').strip("'")
        self.model = model
        self.provider = (os.environ.get("HF_PROVIDER") or provider).strip()
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.list_max_chunks = list_max_chunks
        self.list_max_chars_per_chunk = list_max_chars_per_chunk
        self.temperature = temperature
        self.timeout = timeout

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        analysis: Optional[QueryAnalysis] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderFailure("Hugging Face API key is missing.")

        analysis = analysis or analyze_query(query)
        language = self._detect_language(query)

        if analysis.intent in ("list", "structural"):
            max_chunks = self.list_max_chunks
            max_chars = self.list_max_chars_per_chunk
            max_tokens = 4096
        else:
            max_chunks = self.max_chunks
            max_chars = self.max_chars_per_chunk
            max_tokens = 2048

        llm_chunks = self._prepare_chunks_for_llm(retrieved_chunks, analysis, max_chunks, max_chars)
        context = self._format_context(llm_chunks, language)
        system_prompt = self._build_system_prompt(language, analysis)
        user_prompt = self._build_user_prompt(query, context, language, analysis)

        url = f"https://router.huggingface.co/{self.provider}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        # Only pass enable_thinking if supported by model family
        if "qwen" in self.model.lower():
            payload["enable_thinking"] = False

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if res.status_code != 200:
                raise ProviderFailure(f"HF returned {res.status_code}: {res.text[:200]}")
            data = res.json()
            msg = data["choices"][0]["message"]
            response_text = msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:
            raise ProviderFailure(f"HF request failure: {e}")

        answer, sources = self._parse_response(response_text, retrieved_chunks)
        confidence = self._calculate_confidence(response_text, retrieved_chunks)

        return {
            "answer": answer,
            "language": language,
            "sources": sources,
            "confidence": confidence,
            "provider": "hf",
            "raw_response": response_text,
            "metadata": {},
        }


class HybridGenerator:
    """Orchestrates HF as the primary engine with seamless Ollama failover."""

    def __init__(
        self,
        hf_api_key: Optional[str] = None,
        hf_model: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.1:latest",
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        self.hf_api_key = hf_api_key or os.environ.get("HF_API_KEY", "")
        self.hf_model = hf_model or os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
        self.ollama_host = ollama_host
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

        self.hf_provider = None
        if self.hf_api_key:
            self.hf_provider = HuggingFaceProvider(
                api_key=self.hf_api_key,
                model=self.hf_model,
                temperature=self.temperature,
                **kwargs,
            )
        self.ollama_provider = OllamaGenerator(
            ollama_host=self.ollama_host,
            model=self.model,
            temperature=self.temperature,
            **kwargs,
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        analysis: Optional[QueryAnalysis] = None,
    ) -> Dict[str, Any]:
        provider_failures = []

        # 1. Hugging Face Primary
        if self.hf_provider:
            try:
                res = self.hf_provider.generate(query, retrieved_chunks, analysis)
                if isinstance(res, str):
                    res = {"answer": res, "provider": "hf", "sources": [], "confidence": 0.8, "language": "en", "metadata": {}}
                elif isinstance(res, dict):
                    res.setdefault("metadata", {})
                    if not isinstance(res["metadata"], dict):
                        res["metadata"] = {}
                res["metadata"]["provider_failures"] = provider_failures
                return res
            except ProviderFailure as e:
                logger.warning(f"HF failed: {e}. Falling back to Ollama.")
                provider_failures.append(str(e))

        # 2. Local Ollama Fallback
        try:
            res = self.ollama_provider.generate(query, retrieved_chunks, analysis)
            if isinstance(res, str):
                res = {"answer": res, "provider": "ollama", "sources": [], "confidence": 0.8, "language": "en", "metadata": {}}
            elif isinstance(res, dict):
                res.setdefault("metadata", {})
                if not isinstance(res["metadata"], dict):
                    res["metadata"] = {}
            res["metadata"]["provider_failures"] = provider_failures
            return res
        except ProviderFailure as e:
            logger.error(f"Ollama fallback failed: {e}")
            provider_failures.append(str(e))


        # 3. Graceful Failure
        fallback_sources = [
            {
                "section": c.get("metadata", {}).get("section_num", "Unknown") if isinstance(c.get("metadata"), dict) else "Unknown",
                "source": c.get("source", "Unknown"),
                "excerpt": c.get("text", "")[:300] + "...",
            }
            for c in retrieved_chunks
        ]
        return {
            "answer": "I could not generate an answer from the available policy context.",
            "language": "en",
            "sources": fallback_sources,
            "confidence": 0.0,
            "provider": "error",
            "raw_response": "",
            "metadata": {"provider_failures": provider_failures},
        }