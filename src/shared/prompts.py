"""RAG prompt template shared by all stacks.

Like the chunker, the prompt is a variable held constant: the three stacks
use exactly the same one, so that the comparison is only about the
retrieval strategy.
"""

DEFAULT_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context.
If the context does not contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


def format_contexts(contexts: list[dict]) -> str:
    """Concatenate the retrieved chunks into a numbered context block.

    Each chunk is numbered and, if available, annotated with its source title.
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
    """Assemble the final prompt from the question and the contexts."""
    tmpl = template or DEFAULT_PROMPT_TEMPLATE
    return tmpl.format(context=format_contexts(contexts), question=question)
