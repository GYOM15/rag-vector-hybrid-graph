"""Named entity extraction via spaCy (local NER, en_core_web_sm).

Replaces the old capitalization heuristic with real statistical NER:
we keep only the entity types useful to a knowledge graph
(people, places, organizations, events...), excluding dates and numbers.
The model is loaded lazily and cached.

Model installation: python -m spacy download en_core_web_sm
"""

import re
from functools import lru_cache

# spaCy types kept (we discard DATE, CARDINAL, ORDINAL... = noise for the graph).
_KEEP = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
}
# Leading article sometimes included by spaCy ("The RMS Titanic") — removed for consistent nodes.
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _nlp():
    """Loads en_core_web_sm only once (NER only: tagger/parser disabled)."""
    import spacy

    try:
        return spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
    except OSError as exc:
        raise OSError(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


def extract_entities(text: str, min_length: int = 2) -> list[str]:
    """Named entities of `text` (graph-useful types), deduplicated (case-insensitive)."""
    seen: set[str] = set()
    found: list[str] = []
    for ent in _nlp()(text).ents:
        if ent.label_ not in _KEEP:
            continue
        name = _LEADING_ARTICLE.sub("", ent.text.strip())
        key = name.lower()
        if len(name) < min_length or key in seen:
            continue
        seen.add(key)
        found.append(name)
    return found
