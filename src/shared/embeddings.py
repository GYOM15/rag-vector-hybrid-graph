"""Embedding provider based on sentence-transformers.

sentence-transformers (and torch) is imported lazily, in EmbeddingModel.__init__:
every retriever imports this module for its type hints, and that must stay light.
"""

import re

import numpy as np

_NAME_TOKEN_SEP = re.compile(r"[-_/.]")


def _infer_prefixes(model_name: str) -> tuple[str, str]:
    """Prefixes (query, document) expected by certain model families.

    e5 requires "query: " / "passage: "; bge recommends an instruction on the
    query side. The others (MiniLM, gte...) do not use any. The family is matched
    on whole tokens of the model basename ("intfloat/e5-base-v2" -> {e5, base, v2}),
    not substrings, to avoid false positives.
    """
    basename = model_name.lower().rstrip("/").rsplit("/", 1)[-1]
    tokens = set(_NAME_TOKEN_SEP.split(basename))
    if "e5" in tokens:
        return "query: ", "passage: "
    if "bge" in tokens:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class EmbeddingModel:
    """Wrap sentence-transformers and handle the query/document prefixes.

    The model is selected by `model_name` (all-MiniLM-L6-v2 by default); the
    prefixes are inferred from the name but remain overridable.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        inferred_q, inferred_d = _infer_prefixes(model_name)
        self.query_prefix = inferred_q if query_prefix is None else query_prefix
        self.doc_prefix = inferred_d if doc_prefix is None else doc_prefix
        # `get_sentence_embedding_dimension` was recently renamed `get_embedding_dimension`.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents into vectors, shape (len(texts), dimension)."""
        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into a 1-D vector of shape (dimension,)."""
        return self.model.encode([self.query_prefix + query], convert_to_numpy=True)[0]
