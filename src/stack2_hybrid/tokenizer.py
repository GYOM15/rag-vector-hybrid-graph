"""Tokenisation pour BM25 : minuscule, mots alphanumériques, sans mots vides, racinisés.

Isolé dans son propre module (dépendance légère snowballstemmer) pour rester
testable sans charger faiss/rank_bm25. Nettement meilleur que `lower().split()` :
ponctuation retirée (« 1912. » → « 1912 »), mots vides filtrés, racinisation
Snowball (« diseases »/« disease » → même racine, « plants » → « plant »).
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
    """Tokens BM25 de `text` : minuscule, alphanumérique, sans mots vides, racinisés."""
    words = _WORD.findall(text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    return _STEMMER.stemWords(filtered or words)  # repli sur `words` si tout est vide
