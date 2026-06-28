"""Embedding provider based on sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


def _infer_prefixes(model_name: str) -> tuple[str, str]:
    """Prefixes (query, document) expected by certain model families.

    e5 requires "query: " / "passage: "; bge recommends an instruction on the
    query side. The others (MiniLM, gte...) do not use any.
    """
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name:
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
