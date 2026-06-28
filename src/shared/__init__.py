"""Modules shared by the RAG comparison stacks.

Symbols are exposed *lazily* (PEP 562): importing a lightweight submodule
like ``shared.chunker`` does not pull in the ML stack (embeddings, llm, evaluator).
Each symbol is only loaded on first access via ``shared.<name>`` (or
``from shared import <name>``).
"""

import importlib

# Exposed name -> submodule that defines it.
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
    """Import the relevant submodule on demand (PEP 562)."""
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
