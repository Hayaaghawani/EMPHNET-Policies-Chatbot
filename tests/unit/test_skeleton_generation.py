from src.skeleton_generation import SkeletonLLM


class CapturingLLM(SkeletonLLM):
    def __init__(self):
        pass

    def _complete(self, system, user):
        self.system = system
        self.user = user
        return '{"node_ids":[{"doc_id":"TEST","id":"leaf"}],"broad":false}'


def test_navigation_prompt_contains_compact_memory_only():
    llm = CapturingLLM()
    llm.navigate(
        "What about that in Arabic?",
        [{"doc_id": "TEST", "id": "leaf", "path": "Leave > Annual", "heading": "Annual Leave", "node_type": "leaf"}],
        [],
        [{"question": "What is annual leave?", "nodes": [{"doc_id": "TEST", "id": "leaf", "heading": "Annual Leave"}]}],
    )
    assert 'Q: "What is annual leave?"' in llm.user
    assert "TEST#leaf (Annual Leave)" in llm.user
    assert "own_text" not in llm.user


def test_generation_prompt_does_not_include_history():
    class GenerationCapturingLLM(CapturingLLM):
        def _complete(self, system, user):
            self.system = system
            self.user = user
            return "grounded answer"

    llm = GenerationCapturingLLM()
    result = llm.answer("What is annual leave?", [{"path": "Leave", "text": "Annual leave is provided."}])
    assert result["answer"] == "grounded answer"
    assert "Previous turns" not in llm.user
    assert "Annual leave is provided." in llm.user
