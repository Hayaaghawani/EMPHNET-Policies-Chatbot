from src.generation import HybridGenerator, ProviderFailure


CHUNKS = [{"text": "Sick leave policy text", "source": "Manual • Section 2"}]


class SuccessProvider:
    def __init__(self, answer, provider="hf"):
        self.answer = answer
        self.provider = provider

    def generate(self, query, retrieved_chunks, *_args):
        return {
            "answer": self.answer,
            "provider": self.provider,
            "sources": [c.get("source", "Unknown") for c in retrieved_chunks],
            "confidence": 0.9,
            "language": "en",
            "metadata": {},
        }



class FailingProvider:
    def generate(self, *_args):
        raise ProviderFailure("unavailable")


def _generator():
    return HybridGenerator(hf_api_key=None)


def test_hugging_face_success_uses_consistent_result_contract():
    generator = _generator()
    generator.hf_provider = SuccessProvider("HF answer")
    generator.ollama_provider = FailingProvider()

    result = generator.generate("What is sick leave?", CHUNKS)

    assert result["answer"] == "HF answer"
    assert result["provider"] == "hf"
    assert result["sources"] == ["Manual • Section 2"]
    assert set(result) == {"answer", "provider", "sources", "confidence", "language", "metadata"}


def test_hugging_face_failure_falls_back_to_ollama():
    generator = _generator()
    generator.hf_provider = FailingProvider()
    generator.ollama_provider = SuccessProvider("Ollama answer", provider="ollama")

    result = generator.generate("What is sick leave?", CHUNKS)

    assert result["answer"] == "Ollama answer"
    assert result["provider"] == "ollama"


def test_ollama_success_does_not_require_hugging_face():
    generator = _generator()
    generator.ollama_provider = SuccessProvider("Ollama answer", provider="ollama")

    result = generator.generate("What is sick leave?", CHUNKS)

    assert result["provider"] == "ollama"



def test_both_provider_failures_are_not_successful_answers():
    generator = _generator()
    generator.hf_provider = FailingProvider()
    generator.ollama_provider = FailingProvider()

    result = generator.generate("What is sick leave?", CHUNKS)

    assert result["provider"] == "error"
    assert result["confidence"] == 0.0
    assert result["metadata"]["provider_failures"] == ["unavailable", "unavailable"]
