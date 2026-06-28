"""Tokenization for BM25: lowercase, alphanumeric words, stopwords removed, stemmed.

Isolated in its own module (lightweight snowballstemmer dependency) so it stays
testable without loading faiss/rank_bm25. Clearly better than `lower().split()`:
punctuation stripped ("1912." -> "1912"), stopwords filtered out, Snowball
stemming ("diseases"/"disease" -> same stem, "plants" -> "plant").
"""

import re

from snowballstemmer import stemmer

_WORD = re.compile(r"[a-z0-9]+")
_STEMMER = stemmer("english")
_STOPWORDS = frozenset(
    "a an and the or but if of to in on at by for with from into than then "
    "is are was were be been being it its this that these those he she they we "
    "you i what which who whom when where why how do does did has have had will "
    "would can could should may might must not no as about over under".split()
)


def tokenize(text: str) -> list[str]:
    """BM25 tokens of `text`: lowercase, alphanumeric, stopwords removed, stemmed."""
    words = _WORD.findall(text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    return _STEMMER.stemWords(filtered or words)  # fall back to `words` if everything is empty
