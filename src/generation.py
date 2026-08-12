"""
Generation Module - LLM-based Answer Generation with Grounding

Generates answers strictly from retrieved policy chunks using Ollama LLM.
Enforces grounding: refuses to answer if retrieved chunks don't address the question.
"""

import logging
from typing import List, Dict, Any, Tuple
import requests
import json

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """
    Generates grounded answers using Ollama LLM
    
    Features:
    - Strict grounding to retrieved chunks
    - Language detection (responds in same language as query)
    - Citation formatting with section numbers and excerpts
    - Refusal to answer if chunks don't contain relevant information
    """
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.1:latest",
        temperature: float = 0.3,
        top_p: float = 0.9,
        timeout: int = 300,
        max_chunks: int = 4,
        max_chars_per_chunk: int = 1200,
        num_predict: int = 400,
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.num_predict = num_predict
        
        self._verify_connection()
        self._warmup_model()
    
    def _verify_connection(self) -> None:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available: {model_names}")
            logger.info(f"✓ Connected to Ollama at {self.ollama_host}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_host}. "
                f"Please ensure Ollama is running: ollama serve"
            )

    def _warmup_model(self) -> None:
        """Load model into memory so the first user query is faster."""
        try:
            requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "ok",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=120,
            )
            logger.info(f"Warmed up model: {self.model}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model warmup skipped: {e}")

    @staticmethod
    def _prepare_chunks_for_llm(
        retrieved_chunks: List[Dict[str, Any]],
        max_chunks: int,
        max_chars_per_chunk: int,
    ) -> List[Dict[str, Any]]:
        """Keep only the top chunks and trim text to reduce LLM latency."""
        prepared = []
        for chunk in retrieved_chunks[:max_chunks]:
            trimmed = dict(chunk)
            text = chunk.get("text", "")
            if len(text) > max_chars_per_chunk:
                trimmed["text"] = text[:max_chars_per_chunk] + "\n[...truncated...]"
            prepared.append(trimmed)
        return prepared
    
    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Detect if query is in Arabic or English
        
        Returns: "ar" or "en"
        """
        # Simple heuristic: count Arabic Unicode characters
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_count > len(text) * 0.3 else "en"
    
    def _build_system_prompt(self, language: str) -> str:
        """
        Build system prompt that enforces grounding
        
        Args:
            language: "en" or "ar"
        
        Returns: System prompt string
        """
        if language == "ar":
            return """أنت مساعد متخصص في سياسات الموارد البشرية لمنظمة EMPHNET. أجب بناءً على النصوص المُسترجعة فقط.

قواعد صارمة:
1. اقرأ جميع النصوص المُسترجعة قبل الإجابة.
2. إذا وُجدت الإجابة في أي نص مُسترجع، قدّم كل التفاصيل (أرقام، شروط، خطوات). ممنوع قول "لا أعرف" إذا المعلومة موجودة.
3. اقتبس الأرقام والمدد حرفياً (مثلاً: "14 يوماً").
4. للأسئلة الإجرائية: اذكر الخطوات المرقمة بالترتيب من جميع النصوص ذات الصلة.
5. لا تختلق معلومات غير موجودة في النصوص.
6. الرد بالعربية.

التنسيق:
- إجابة مباشرة وواضحة مع كل التفاصيل
- للإجراءات: قائمة خطوات مرقمة
- اقتباس قصير من النص المصدر
- مرجع القسم/الإجراء"""
        else:
            return """You are an EMPHNET HR policy assistant. Answer ONLY from the retrieved text.

Rules:
1. Use ALL details from the context: numbers, eligibility, steps.
2. Never say "not found" if the answer is in the context.
3. For how-to questions, list numbered steps from all relevant chunks.
4. Do not invent information.

Format: direct answer, then brief quote and source reference."""
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer based on query and retrieved chunks
        
        Args:
            query: User question
            retrieved_chunks: List of dicts with 'text' and 'metadata' keys
                from retriever.retrieve()
        
        Returns:
            Dict with keys:
            - answer: Generated response text
            - language: Language of the response
            - sources: List of source citations
            - confidence: Confidence score (0-1)
        """
        # Detect language
        language = self._detect_language(query)
        logger.info(f"Detected language: {language}")

        llm_chunks = self._prepare_chunks_for_llm(
            retrieved_chunks, self.max_chunks, self.max_chars_per_chunk
        )
        context = self._format_context(llm_chunks, language)
        logger.info(
            f"LLM context: {len(llm_chunks)} chunks, {len(context)} chars"
        )
        
        # Build prompt
        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(query, context, language)
        
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"User prompt length: {len(user_prompt)} chars")
        
        # Call Ollama
        response_text = self._call_ollama(system_prompt, user_prompt)
        
        # Parse response and extract sources
        answer, sources = self._parse_response(response_text, retrieved_chunks)
        
        # Calculate confidence (simple heuristic)
        confidence = self._calculate_confidence(response_text, retrieved_chunks)
        
        return {
            "answer": answer,
            "language": language,
            "sources": sources,
            "confidence": confidence,
            "raw_response": response_text
        }
    
    @staticmethod
    def _format_context(
        retrieved_chunks: List[Dict[str, Any]],
        language: str
    ) -> str:
        """Format retrieved chunks into context string"""
        if language == "ar":
            header = "المستندات ذات الصلة:\n"
            separator = "\n---\n"
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
    
    def _build_user_prompt(
        self,
        query: str,
        context: str,
        language: str
    ) -> str:
        """Build user prompt with query and context"""
        if language == "ar":
            return f"""السؤال: {query}

السياق:
{context}

الرجاء الإجابة على السؤال بناءً على السياق المقدم أعلاه فقط."""
        else:
            return f"""Question: {query}

Context:
{context}

Answer the question using ONLY the context above. Synthesize information from all relevant chunks. For how-to questions, list all steps in order."""
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama API and return response"""
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
                "num_predict": self.num_predict,
                "num_ctx": 4096,
            },
        }
        
        logger.info(f"Calling Ollama ({self.model}), timeout={self.timeout}s")
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.Timeout:
            return (
                "The model is taking too long on this machine. "
                "Try a smaller model (e.g. llama3.1:latest) in your .env file, "
                "or ask a shorter question."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error communicating with LLM: {str(e)}"
    
    @staticmethod
    def _parse_response(
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Parse response and extract sources
        
        Returns:
            (answer_text, sources_list)
        """
        # Simple parsing: everything before "Source:" is the answer
        parts = response_text.split("[Source:")
        answer = parts[0].strip()
        
        # Extract sources from retrieved chunks
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
                "excerpt": text_excerpt
            })
        
        return answer, sources
    
    @staticmethod
    def _calculate_confidence(
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score (0-1)
        
        Heuristics:
        - Lower if response contains "I couldn't find", "not found", etc.
        - Lower if only 1-2 chunks retrieved
        - Higher if multiple high-quality chunks
        """
        confidence = 0.7  # Base confidence
        
        # Reduce if refusal phrases present
        refusal_phrases = [
            "couldn't find",
            "not found",
            "not mentioned",
            "no information",
            "unable to find"
        ]
        if any(phrase in response_text.lower() for phrase in refusal_phrases):
            confidence -= 0.3
        
        # Adjust based on number of chunks
        num_chunks = len(retrieved_chunks)
        if num_chunks < 2:
            confidence -= 0.2
        elif num_chunks >= 5:
            confidence += 0.1
        
        # Adjust based on average score
        if retrieved_chunks:
            avg_score = sum(c.get('score', 0) for c in retrieved_chunks) / len(retrieved_chunks)
            confidence += (avg_score - 0.5) * 0.2
        
        return max(0, min(1, confidence))  # Clamp to [0, 1]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = OllamaGenerator(
        ollama_host="http://localhost:11434",
        model="qwen2.5:14b-instruct"
    )
    
    # Example retrieved chunks (mock)
    mock_chunks = [
        {
            "text": "Sick leave is provided to employees for medical reasons. "
                   "Employees must provide a medical certificate after 3 consecutive days.",
            "metadata": {"section_num": 2, "page_number": 45},
            "source": "HR Manual • Section 2.4 • p. 45"
        }
    ]
    
    # Generate answer
    query = "What is the sick leave policy?"
    result = generator.generate(query, mock_chunks)
    
    print(f"Answer: {result['answer']}\n")
    print(f"Confidence: {result['confidence']:.2f}\n")
    print(f"Sources: {json.dumps(result['sources'], indent=2)}")
