"""Unit tests for ``recursive_chunk`` (src/shared/chunker.py).

These tests pin down the contract of the recursive character chunker so that
the *chunking variable* stays correct and identical across all three RAG
stacks. They focus on the behaviours that are easy to get subtly wrong:
size bounds, overlap, the separator hierarchy, the no-separator fallback,
metadata propagation, and indexing.
"""

import pytest

from chunker import recursive_chunk, Chunk


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

PARAGRAPH = (
    "Retrieval augmented generation combines a retriever with a generator. "
    "The retriever finds the most relevant chunks for a query. "
    "The generator then conditions its answer on those chunks. "
    "This grounds the response in real sources and reduces hallucination."
)

# ~10 paragraphs separated by blank lines -> a few thousand characters.
LONG_TEXT = "\n\n".join(f"Section {i}. {PARAGRAPH}" for i in range(10))


def _lengths(chunks: list[Chunk]) -> list[int]:
    return [len(c.text) for c in chunks]


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

def test_short_text_returns_single_chunk():
    text = "A short sentence that fits entirely."
    chunks = recursive_chunk(text, max_size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == text  # already stripped, fits in one chunk
    assert chunks[0].index == 0
    assert chunks[0].metadata["chunk_strategy"] == "recursive"


@pytest.mark.parametrize("text", ["", "   ", "\n\n  \t\n"])
def test_empty_or_whitespace_returns_no_chunks(text):
    assert recursive_chunk(text) == []


def test_long_text_produces_multiple_chunks():
    chunks = recursive_chunk(LONG_TEXT, max_size=300, overlap=50)
    assert len(chunks) > 1


def test_no_chunk_is_empty():
    chunks = recursive_chunk(LONG_TEXT, max_size=200, overlap=40)
    assert all(c.text.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Size bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_size", [100, 200, 300, 500])
def test_no_overlap_respects_max_size_strictly(max_size):
    """With overlap=0 every chunk must fit within max_size."""
    chunks = recursive_chunk(LONG_TEXT, max_size=max_size, overlap=0)
    assert chunks, "expected at least one chunk"
    assert max(_lengths(chunks)) <= max_size


@pytest.mark.parametrize("max_size,overlap", [(100, 20), (200, 40), (300, 50)])
def test_overlap_stays_within_documented_bound(max_size, overlap):
    """With overlap>0 a chunk may exceed max_size, but only by the prepended
    overlap plus one join separator (<= 2 chars). This documents the real
    behaviour: chunks are NOT strictly <= max_size once overlap is on."""
    chunks = recursive_chunk(LONG_TEXT, max_size=max_size, overlap=overlap)
    assert chunks
    assert max(_lengths(chunks)) <= max_size + overlap + 2


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------

def test_overlap_adds_duplicated_content():
    """Turning overlap on must increase the total character count, because the
    tail of each chunk is repeated at the head of the next one."""
    with_overlap = recursive_chunk(LONG_TEXT, max_size=300, overlap=60)
    without_overlap = recursive_chunk(LONG_TEXT, max_size=300, overlap=0)

    total_with = sum(_lengths(with_overlap))
    total_without = sum(_lengths(without_overlap))
    assert total_with > total_without


def test_hard_split_overlap_is_exact():
    """A string with no usable separator hits the character-level fallback,
    where overlap is exact and easy to verify. With max_size=100, overlap=20
    the window advances by 80 chars each step."""
    text = "x" * 1000
    chunks = recursive_chunk(text, max_size=100, overlap=20)

    starts = list(range(0, 1000, 80))  # 0, 80, ..., 960
    assert len(chunks) == len(starts)
    # Every chunk but the last is a full 100-char window.
    assert all(len(c.text) == 100 for c in chunks[:-1])
    # The last window is the remainder: text[960:1000] -> 40 chars.
    assert len(chunks[-1].text) == 40


# ---------------------------------------------------------------------------
# Degenerate configs must be handled, not crash (defensive clamping)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overlap", [100, 150, 10_000])
def test_overlap_geq_max_size_does_not_crash_or_drop_text(overlap):
    """overlap >= max_size used to raise ValueError (overlap == max_size) or
    silently return zero chunks (overlap > max_size) on separator-less text.
    Now it must degrade gracefully: valid, non-empty chunks, no data loss."""
    text = "".join(str(i % 10) for i in range(500))  # 500 chars, no separators
    chunks = recursive_chunk(text, max_size=100, overlap=overlap)

    assert chunks, "must still produce chunks"
    assert all(c.text for c in chunks)
    assert text.startswith(chunks[0].text)        # begins at the start
    assert text.endswith(chunks[-1].text)         # reaches the end
    assert set("".join(c.text for c in chunks)) == set(text)  # full coverage


def test_non_positive_max_size_is_clamped():
    """max_size <= 0 must not crash; it is clamped up to a usable minimum."""
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = recursive_chunk(text, max_size=0, overlap=50)
    assert chunks
    assert all(c.text for c in chunks)


# ---------------------------------------------------------------------------
# Separator hierarchy
# ---------------------------------------------------------------------------

def test_splits_on_paragraph_boundary_first():
    """Two short paragraphs that together exceed max_size should split cleanly
    on the blank line, yielding one chunk per paragraph (no overlap)."""
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = recursive_chunk(text, max_size=30, overlap=0)

    assert len(chunks) == 2
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text


def test_custom_separators_are_honoured():
    text = "alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel"
    chunks = recursive_chunk(text, max_size=20, overlap=0, separators=["|"])

    assert len(chunks) > 1
    # The pipe is the join separator, so tokens are recombined with '|'.
    joined = "".join(c.text for c in chunks)
    for token in ("alpha", "bravo", "hotel"):
        assert token in joined


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_indices_are_sequential():
    chunks = recursive_chunk(LONG_TEXT, max_size=250, overlap=40)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_is_propagated_to_every_chunk():
    meta = {"source": "doc-1", "url": "http://example.com"}
    chunks = recursive_chunk(LONG_TEXT, max_size=300, overlap=50, metadata=meta)

    assert chunks
    for c in chunks:
        assert c.metadata["source"] == "doc-1"
        assert c.metadata["url"] == "http://example.com"
        assert c.metadata["chunk_strategy"] == "recursive"


def test_metadata_argument_is_not_mutated():
    """The caller's dict must not be modified (each chunk gets a copy)."""
    meta = {"source": "doc-1"}
    recursive_chunk(LONG_TEXT, max_size=300, overlap=50, metadata=meta)
    assert meta == {"source": "doc-1"}  # no chunk_strategy leaked back in


def test_chunks_are_independent_copies_of_metadata():
    """Mutating one chunk's metadata must not affect the others."""
    chunks = recursive_chunk(LONG_TEXT, max_size=300, overlap=50,
                             metadata={"source": "doc-1"})
    assert len(chunks) > 1
    chunks[0].metadata["source"] = "MUTATED"
    assert chunks[1].metadata["source"] == "doc-1"
