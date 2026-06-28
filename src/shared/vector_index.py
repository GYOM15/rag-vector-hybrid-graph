"""FAISS vector indexer (IndexFlatIP + L2 normalization = cosine similarity).

Stores the chunks and their metadata, aligned by position. Persistence via
save/load (.faiss / .chunks.json / .meta.json).
"""

import json
from pathlib import Path

import faiss
import numpy as np


class FaissIndexer:
    """FAISS index shared by the stacks: exact cosine search over the chunks."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[str], metadata: list[dict]):
        """Add chunks and their embeddings (L2-normalized before insertion)."""
        if len(embeddings) != len(chunks) or len(embeddings) != len(metadata):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings, "
                f"{len(chunks)} chunks, {len(metadata)} metadata entries."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )

        # L2 normalization: cosine similarity is then obtained via the dot product.
        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def save(self, path: str):
        """Persist the index to disk: {path}.faiss, .chunks.json, .meta.json."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(base.with_suffix(".faiss")))

        with open(base.with_suffix(".chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        with open(base.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, path: str):
        """Reload a saved index (same prefix as save())."""
        base = Path(path)

        index_path = base.with_suffix(".faiss")
        chunks_path = base.with_suffix(".chunks.json")
        meta_path = base.with_suffix(".meta.json")

        for p in (index_path, chunks_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
