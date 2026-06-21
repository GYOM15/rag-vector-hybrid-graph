"""Pytest configuration for the test suite.

We add ``src/shared`` directly to ``sys.path`` so tests can ``import chunker``
without going through ``shared/__init__.py`` -- that package init eagerly
imports embeddings/llm/evaluator, which pull in sentence-transformers, faiss,
etc. The chunker only depends on the standard library, so importing it in
isolation keeps the tests fast and dependency-free.
"""

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent / "src" / "shared"
sys.path.insert(0, str(_SHARED))
