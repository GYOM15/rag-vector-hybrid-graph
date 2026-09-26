"""*Judge-free* answer metrics: Exact Match and F1 (SQuAD/HotpotQA style).

Deterministic, pure stdlib. They measure the quality of a generated answer against
a *gold* answer, without a judge model or API key -- to close the
"better retrieval -> better answer" loop in a reproducible way.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, no punctuation or articles, collapsed whitespace."""
    text = (text or "").lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 if the normalized answers are identical, otherwise 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 (SQuAD style): overlap between prediction and gold."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)  # 1.0 if both empty, otherwise 0.0
    n_same = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if not n_same:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str, gold: str) -> float:
    """1.0 if the normalized gold occurs in the normalized prediction as whole tokens.

    Both sides go through `normalize_answer`, then the gold's token sequence must
    appear *contiguously* in the prediction's tokens ("the capital is Kabul" contains
    "Kabul"; "Kabuli" does not, nor does "Kabul ... city" contain "Kabul city" with a
    gap). Unlike EM/F1 it does not penalize a correct but verbose answer, which is what
    small instruct models produce even when asked to be brief. It rewards hedging that
    lists several candidates, so it is reported *alongside* EM/F1, never instead.
    An empty gold contains nothing (0.0).
    """
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    n = len(gold_toks)
    if not n:
        return 0.0
    return float(any(pred_toks[i:i + n] == gold_toks for i in range(len(pred_toks) - n + 1)))
