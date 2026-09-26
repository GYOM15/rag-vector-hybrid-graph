"""Pytest configuration for the test suite.

We add ``src/shared`` directly to ``sys.path`` so tests can ``import chunker``
without going through ``shared/__init__.py`` -- that package init eagerly
imports embeddings/llm/evaluator, which pull in sentence-transformers, faiss,
etc. The chunker only depends on the standard library, so importing it in
isolation keeps the tests fast and dependency-free.

``src`` itself goes *first* on ``sys.path`` so that package imports
(``shared.vector_index``, ``stack2_hybrid.retriever``...) resolve to this
checkout, even when another checkout of the project is installed in editable
mode in the environment.
"""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_SHARED = _SRC / "shared"
sys.path.insert(0, str(_SHARED))
sys.path.insert(0, str(_SRC))

# macOS wheels of faiss-cpu and torch each bundle their own libomp. The retriever
# tests start faiss's OpenMP runtime before the NER tests import spaCy (thinc then
# imports torch, when installed), and the second runtime aborts the whole process.
# Workaround for the test process only (the eval entry points are not affected).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
