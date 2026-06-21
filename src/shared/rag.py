"""Squelette RAG partagé par tous les stacks.

Le pipeline est identique pour les trois architectures : récupérer les chunks,
assembler le prompt, appeler le LLM. Seule la stratégie de récupération change
d'un stack à l'autre. On factorise donc l'orchestration ici ; un stack ne
fournit que son `retriever` (un objet exposant `search(query, k) -> list[dict]`).
"""

import time
from typing import Callable, Protocol

from .prompts import DEFAULT_PROMPT_TEMPLATE, build_prompt


class Retriever(Protocol):
    """Interface commune des récupérateurs : renvoyer les k chunks pertinents."""

    def search(self, query: str, k: int = 5) -> list[dict]:
        ...


class BaseRAG:
    """Pipeline RAG générique : récupération → prompt → génération, avec latences."""

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
        """Exécute le pipeline. Renvoie {answer, contexts, retrieval_ms, generation_ms, latency_ms}."""
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
