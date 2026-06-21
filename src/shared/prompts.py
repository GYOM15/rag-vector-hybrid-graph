"""Gabarit de prompt RAG partagé par tous les stacks.

Comme le chunker, le prompt est une variable tenue constante : les trois stacks
utilisent exactement le même, pour que la comparaison ne porte que sur la
stratégie de récupération.
"""

DEFAULT_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context.
If the context does not contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


def format_contexts(contexts: list[dict]) -> str:
    """Concatène les chunks récupérés en un bloc de contexte numéroté.

    Chaque chunk est numéroté et, si disponible, annoté de son titre source.
    """
    parts = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx["metadata"].get("title", "")
        header = f"[{i}] (source: {title})" if title else f"[{i}]"
        parts.append(f"{header}\n{ctx['text']}")
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    contexts: list[dict],
    template: str | None = None,
) -> str:
    """Assemble le prompt final à partir de la question et des contextes."""
    tmpl = template or DEFAULT_PROMPT_TEMPLATE
    return tmpl.format(context=format_contexts(contexts), question=question)
