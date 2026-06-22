"""Extraction d'entités nommées via spaCy (NER local, en_core_web_sm).

Remplace l'ancienne heuristique de capitalisation par un vrai NER statistique :
on ne garde que les types d'entités utiles à un graphe de connaissances
(personnes, lieux, organisations, événements…), en excluant dates et nombres.
Le modèle est chargé paresseusement et mis en cache.

Installation du modèle : python -m spacy download en_core_web_sm
"""

import re
from functools import lru_cache

# Types spaCy conservés (on écarte DATE, CARDINAL, ORDINAL… = bruit pour le graphe).
_KEEP = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
}
# Article de tête parfois inclus par spaCy (« The RMS Titanic ») — retiré pour des nœuds cohérents.
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _nlp():
    """Charge en_core_web_sm une seule fois (NER seul : tagger/parser désactivés)."""
    import spacy

    try:
        return spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
    except OSError as exc:
        raise OSError(
            "Modèle spaCy 'en_core_web_sm' introuvable. Installe-le avec :\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


def extract_entities(text: str, min_length: int = 2) -> list[str]:
    """Entités nommées de `text` (types utiles au graphe), dédupliquées (casse ignorée)."""
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
