"""Tokenization for BM25: lowercase, alphanumeric words, stopwords removed, stemmed.

Isolated in its own module (lightweight snowballstemmer dependency) so it stays
testable without loading faiss/rank_bm25. Clearly better than `lower().split()`:
punctuation stripped ("1912." -> "1912"), stopwords filtered out, Snowball
stemming ("diseases"/"disease" -> same stem, "plants" -> "plant").
Words are Unicode letters/digits plus combining marks (NFC-normalized): an
ASCII-only pattern would cut "café" into a bogus "caf" token, and Python's regex
word characters alone cut "हिन्दी" at every vowel sign (combining marks are not
word characters).
"""

import re
import sys
import unicodedata

from snowballstemmer import stemmer


def _combining_marks_class() -> str:
    """Regex class body (as ranges) of the Unicode combining marks (Mn, Mc, Me)."""
    ranges: list[list[int]] = []
    for code in range(sys.maxunicode + 1):
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    return "".join(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}" for lo, hi in ranges)


# A letter or digit (\w without the underscore), then letters, digits and combining
# marks (unrolled loop). A mark with no letter before it starts no word.
_WORD = re.compile(rf"[^\W_]+(?:[{_combining_marks_class()}]+[^\W_]*)*")
_STEMMER = stemmer("english")
_STOPWORDS = frozenset(
    "a an and the or but if of to in on at by for with from into than then "
    "is are was were be been being it its this that these those he she they we "
    "you i what which who whom when where why how do does did has have had will "
    "would can could should may might must not no as about over under".split()
)


def tokenize(text: str) -> list[str]:
    """BM25 tokens of `text`: lowercase, alphanumeric, stopwords removed, stemmed."""
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    return _STEMMER.stemWords(filtered or words)  # fall back to `words` if everything is empty
