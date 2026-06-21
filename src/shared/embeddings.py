"""Fournisseur d'embeddings basé sur sentence-transformers."""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Encapsule sentence-transformers pour une génération d'embeddings cohérente.

    Utilise all-MiniLM-L6-v2 par défaut (384 dimensions, rapide, bonne qualité).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # `get_sentence_embedding_dimension` est renommé `get_embedding_dimension`
        # dans les versions récentes : on utilise le nouveau nom s'il existe.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode une liste de textes en vecteurs, forme (len(texts), dimension)."""
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode une seule requête en un vecteur 1-D de forme (dimension,)."""
        return self.model.encode([query], convert_to_numpy=True)[0]
