"""Modules partagés par les stacks de comparaison RAG.

Les symboles sont exposés en *lazy* (PEP 562) : importer un sous-module léger
comme ``shared.chunker`` ne tire pas la pile ML (embeddings, llm, evaluator).
Chaque symbole n'est chargé qu'au premier accès via ``shared.<nom>`` (ou
``from shared import <nom>``).
"""

import importlib

# Nom exposé -> sous-module qui le définit.
_EXPORTS = {
    "Chunk": ".chunker",
    "recursive_chunk": ".chunker",
    "EmbeddingModel": ".embeddings",
    "FaissIndexer": ".vector_index",
    "call_llm": ".llm",
    "evaluate_rag": ".evaluator",
    "DEFAULT_PROMPT_TEMPLATE": ".prompts",
    "build_prompt": ".prompts",
    "format_contexts": ".prompts",
    "BaseRAG": ".rag",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Importe le sous-module concerné à la demande (PEP 562)."""
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
