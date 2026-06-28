"""RAG skeleton shared by all stacks.

The pipeline is identical for the three architectures: retrieve the chunks,
assemble the prompt, call the LLM. Only the retrieval strategy changes
from one stack to another. So we factor the orchestration out here; a stack
only provides its `retriever` (an object exposing `search(query, k) -> list[dict]`).
"""

import time
from typing import Callable, Protocol

from .prompts import DEFAULT_PROMPT_TEMPLATE, build_prompt


class Retriever(Protocol):
    """Common retriever interface: return the k relevant chunks."""

    def search(self, query: str, k: int = 5) -> list[dict]:
        ...


class BaseRAG:
    """Generic RAG pipeline: retrieval -> prompt -> generation, with latencies."""

    def __init__(
        self,
        retriever: Retriever,
        llm_fn: Callable[[str], str],
        prompt_template: str | None = None,
    ):
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    def query(self, question: str, k: int = 5) -> dict:
        """Run the pipeline. Returns {answer, contexts, retrieval_ms, generation_ms, latency_ms}."""
        start = time.perf_counter()
        contexts = self.retriever.search(question, k=k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        prompt = build_prompt(question, contexts, self.prompt_template)

        gen_start = time.perf_counter()
        answer = self.llm_fn(prompt)
        generation_ms = (time.perf_counter() - gen_start) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
