"""Métriques de réponse *sans juge* : Exact Match et F1 (style SQuAD/HotpotQA).

Déterministes, pur stdlib. Mesurent la qualité d'une réponse générée contre une
réponse *gold*, sans modèle-juge ni clé d'API — pour fermer la boucle
« meilleure récupération → meilleure réponse » de façon reproductible.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalisation SQuAD : minuscules, sans ponctuation ni articles, espaces réduits."""
    text = (text or "").lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 si les réponses normalisées sont identiques, sinon 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """F1 au niveau des tokens (style SQuAD) : recouvrement entre prédiction et gold."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)  # 1.0 si les deux vides, sinon 0.0
    n_same = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if not n_same:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
