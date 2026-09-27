"""Tests for the shared RAG skeleton and the benchmark loop — no model, no index.

The retriever, the generator and RAGAS are faked: we check the wiring the app relies
on (per-call generator override, RAGAS failures reported instead of swallowed).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared import evaluator  # noqa: E402
from shared.rag import BaseRAG  # noqa: E402


class _FakeRetriever:
    def search(self, query, k=5):
        return [{"text": f"chunk about {query}", "metadata": {"title": "T"}}][:k]


# ---------------------------------------------------------------------------
# BaseRAG.query(llm_fn=...)
# ---------------------------------------------------------------------------

def test_query_uses_the_stack_generator_by_default():
    rag = BaseRAG(_FakeRetriever(), lambda prompt: "default")
    assert rag.query("q")["answer"] == "default"


def test_llm_fn_overrides_the_generator_for_one_call():
    default_prompts, session_prompts = [], []
    rag = BaseRAG(_FakeRetriever(), lambda p: default_prompts.append(p) or "default")

    result = rag.query("Who?", k=1, llm_fn=lambda p: session_prompts.append(p) or "session")

    assert result["answer"] == "session"
    assert default_prompts == []                      # the stack's generator wasn't called
    assert "chunk about Who?" in session_prompts[0]   # ...the override got the full prompt
    assert "Question: Who?" in session_prompts[0]
    assert rag.query("Who?")["answer"] == "default"   # the (shared) stack is unchanged


# ---------------------------------------------------------------------------
# evaluate_stacks
# ---------------------------------------------------------------------------

class _FakeRAG:
    def __init__(self):
        self.llm_fns = []

    def query(self, question, k=5, llm_fn=None):
        self.llm_fns.append(llm_fn)
        return {"answer": "a", "contexts": [{"text": "c"}],
                "retrieval_ms": 1.0, "generation_ms": 2.0, "latency_ms": 3.0}


def test_empty_question_list_raises():
    with pytest.raises(ValueError):
        evaluator.evaluate_stacks({"S": _FakeRAG()}, [], [])


def test_ragas_failure_is_recorded_not_swallowed(monkeypatch):
    def unreachable_judge(*args, **kwargs):
        raise RuntimeError("judge unreachable")

    monkeypatch.setattr(evaluator, "evaluate_rag", unreachable_judge)
    rag, session_llm = _FakeRAG(), (lambda prompt: "x")

    out = evaluator.evaluate_stacks({"S": rag}, ["q1", "q2"], ["g1", "g2"],
                                    types=["a", "b"], llm_fn=session_llm)

    assert out["S"]["ragas_error"] == "RuntimeError: judge unreachable"
    assert out["S"]["avg_latency_ms"] == 3.0                        # latencies still there
    assert out["S"]["by_type"]["a"] == {"n": 1, "avg_latency_ms": 3.0}
    assert rag.llm_fns == [session_llm, session_llm]                # generator forwarded
