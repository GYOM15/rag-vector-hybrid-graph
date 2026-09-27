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


# A token next to one of these turns the answer into a choice: "2001 or 2002", "A vs B".
_ALTERNATIVES = frozenset({"or", "nor", "versus", "vs"})
_POLAR = {"yes": "no", "no": "yes"}  # yes/no gold -> the opposite polarity
# A yes/no reply must open with the word standing alone ("No, ...", "Yes.", "no"),
# not "No one knows", "No information in the context" or "Yes/no".
_LEADING_POLAR = re.compile(r"^\W*(yes|no)\b\s*(?:[^\w\s/]|$)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?;\n]")


def _occurrences(toks: list[str], sub: list[str]) -> list[int]:
    """Start indices where `sub` occurs contiguously in `toks`."""
    n = len(sub)
    return [i for i in range(len(toks) - n + 1) if toks[i:i + n] == sub]


def _polar_answer(pred: str, gold: str) -> float:
    """Yes/no gold: the reply opens with the gold word, standing alone, and its first
    sentence does not also say the opposite ("Yes, and no")."""
    lead = _LEADING_POLAR.match(pred or "")
    if not lead or lead.group(1).lower() != gold:
        return 0.0
    first_sentence = _SENTENCE_END.split(pred.strip(), maxsplit=1)[0]
    return float(_POLAR[gold] not in normalize_answer(first_sentence).split())


def contains_answer(pred: str, gold: str, question: str = "") -> float:
    """1.0 if the normalized prediction contains the normalized gold as an *answer*.

    Unlike EM/F1 it does not penalize a correct but verbose answer, which is what
    small instruct models produce even when asked to be brief. But raw containment
    also credits wrong answers in HotpotQA's most common shapes, so it is guarded:

    - general case: the gold's tokens (after `normalize_answer`) occur *contiguously*
      in the prediction's tokens ("the capital is Kabul" contains "Kabul"; "Kabuli"
      does not, nor does "Kabul ... city" contain "Kabul city"), in at least one place
      not adjacent to an alternative ("or", "nor", "vs", "versus"): "2001 or 2002"
      and "either 1999 or 2001" credit neither year;
    - yes/no golds: the reply must *open* with the gold word standing alone, and its
      first sentence must not contain the opposite polarity. "No, ..." credits "no";
      a refusal that merely mentions the word ("I do not know; there is no
      information", "No information in the context") or a hedge ("Yes and no") does not;
    - golds that occur in the `question` (choice questions: "Who is older, Annie Morton
      or Terry Richardson?"): restating the question contains the gold whichever option
      the model picks, so the reply must *open* with the gold and not continue with an
      alternative or a second option ("Terry Richardson or Annie Morton",
      "Letters to Cleo and Screaming Trees" score 0).

    Remaining limit: a comma list of candidates ("1999, 2000, 2001") still contains
    the gold. So contains is reported *alongside* EM/F1, never instead, and next to the
    mean answer length, which exposes that kind of hedging. The guards are
    conservative: correct answers phrased "..., so the answer is yes" or "The older one
    is Terry Richardson" score 0 (they still get partial F1).
    An empty gold contains nothing (0.0).
    """
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    n = len(gold_toks)
    if not n:
        return 0.0
    if gold_toks[0] in _POLAR and n == 1:
        return _polar_answer(pred, gold_toks[0])
    if _occurrences(normalize_answer(question).split(), gold_toks):
        after = pred_toks[n] if len(pred_toks) > n else None
        return float(pred_toks[:n] == gold_toks and after not in _ALTERNATIVES | {"and"})
    return float(any(
        (i == 0 or pred_toks[i - 1] not in _ALTERNATIVES)
        and (i + n == len(pred_toks) or pred_toks[i + n] not in _ALTERNATIVES)
        for i in _occurrences(pred_toks, gold_toks)))
