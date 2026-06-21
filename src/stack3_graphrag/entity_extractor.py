"""Extraction d'entités légère, sans dépendance lourde.

Heuristique : on repère les suites de mots capitalisés (noms propres probables)
comme entités, en retirant les mots vides capitalisés en début de phrase.
Suffisant pour démontrer le graphe ; un vrai NER (spaCy) ou un LLM donnerait de
meilleurs résultats — c'est un remplacement facile derrière la même interface.
"""

import re

# Mots capitalisés courants (souvent en début de phrase) à ne pas prendre pour des entités.
_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "You", "In", "On", "At", "Of", "For", "And", "But", "Or",
    "If", "When", "While", "After", "Before", "However", "There", "Here", "As",
}

_CAPITALIZED_SEQUENCE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


def extract_entities(text: str, min_length: int = 3) -> list[str]:
    """Renvoie la liste dédupliquée des entités détectées dans `text`.

    Une entité est une suite de mots capitalisés ; on retire d'éventuels mots
    vides en tête (ex. « The » dans « The Titanic ») et on ignore les entités
    plus courtes que `min_length`. La déduplication est insensible à la casse.
    """
    found: list[str] = []
    seen: set[str] = set()

    for match in _CAPITALIZED_SEQUENCE.finditer(text):
        words = match.group(1).split()
        while words and words[0] in _STOPWORDS:
            words = words[1:]
        if not words:
            continue

        entity = " ".join(words)
        if len(entity) < min_length:
            continue

        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(entity)

    return found
