"""Recursive text splitting with overlap for RAG systems.

Splits a text along a hierarchy of separators (paragraph, line,
sentence, then word) and applies a configurable character overlap in order
to preserve context at chunk boundaries.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A piece of text with its metadata and position index."""

    text: str
    metadata: dict = field(default_factory=dict)
    index: int = 0


_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _make_chunk(text: str, index: int, metadata: dict | None) -> Chunk:
    """Build a Chunk with a copy of the metadata and the strategy."""
    meta = dict(metadata) if metadata else {}
    meta["chunk_strategy"] = "recursive"
    return Chunk(text=text, metadata=meta, index=index)


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split `text` on `separator`, ignoring empty fragments."""
    if separator == "":
        return list(text)  # last resort: character by character
    return [part for part in text.split(separator) if part.strip()]


def _split_into_windows(text: str, size: int, step: int) -> list[str]:
    """Split `text` into windows of `size` characters advancing by `step`."""
    return [
        window
        for start in range(0, len(text), step)
        if (window := text[start:start + size].strip())
    ]


# ---------------------------------------------------------------------------
# Steps of the recursive splitting
# ---------------------------------------------------------------------------

def _normalize_segments(
    segments: list[str],
    size: int, 
    remaining_separators: list[str],
) -> list[str]:
    """Guarantee that each segment fits within `size`.

    A segment that is too large is re-split using the remaining separators, or
    failing that into character windows. Overlap is not applied here:
    it is applied at assembly time, on the final list of segments.
    """
    result: list[str] = []
    for seg in segments:
        if len(seg) <= size:
            result.append(seg.strip())
        elif remaining_separators:
            sub = _recursive_split(seg, size, 0, remaining_separators, None)
            result.extend(chunk.text for chunk in sub)
        else:
            result.extend(_split_into_windows(seg, size, size))
    return result


def _build_chunks_with_overlap(
    segments: list[str],
    size: int,
    overlap: int,
    join_sep: str,
    metadata: dict | None,
) -> list[Chunk]:
    """Group the segments into chunks of at most `size` characters.

    The last `overlap` characters of a chunk are repeated at the start of the
    next one to preserve context.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for seg in segments:
        added = (len(join_sep) if current else 0) + len(seg)
        if current and current_len + added > size:
            # The segment overflows: we close the current chunk, then start the
            # next one with the end of the previous one (the overlap).
            text = join_sep.join(current)
            chunks.append(_make_chunk(text, len(chunks), metadata))
            tail = text[-overlap:] if overlap else ""
            current = [tail] if tail else []
            current_len = len(tail)
            added = (len(join_sep) if current else 0) + len(seg)
        current.append(seg)
        current_len += added

    if current:
        chunks.append(_make_chunk(join_sep.join(current), len(chunks), metadata))
    return chunks


def _recursive_split(
    text: str,
    size: int,
    overlap: int,
    separators: list[str],
    metadata: dict | None,
) -> list[Chunk]:
    """Pick the first separator that splits, normalize, then assemble."""
    if len(text) <= size:
        stripped = text.strip()
        return [_make_chunk(stripped, 0, metadata)] if stripped else []

    for i, sep in enumerate(separators):
        segments = _split_by_separator(text, sep)
        if len(segments) <= 1:
            continue  # this separator splits nothing: move on to the next one
        fine = _normalize_segments(segments, size, separators[i + 1:])
        return _build_chunks_with_overlap(fine, size, overlap, sep, metadata)

    # No separator worked: overlapping character windows.
    # `step` is clamped to 1 to never loop indefinitely or lose text.
    windows = _split_into_windows(text, size, max(size - overlap, 1))
    return [_make_chunk(window, idx, metadata) for idx, window in enumerate(windows)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recursive_chunk(
    text: str,
    max_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split `text` recursively along a hierarchy of separators.

    Tries to cut by paragraph, then line, then sentence, then word. Each
    chunk is at most `max_size` characters (excluding the overlap carried over
    from the previous chunk), and `overlap` trailing characters are repeated at the start of the next one.
    `metadata` is copied onto each returned chunk.
    """
    if separators is None:
        separators = _DEFAULT_SEPARATORS

    # Defensive normalization: a caller may provide overlap >= max_size or
    # a max_size <= 0, which would make the splitting step <= 0 (crash or
    # text loss). We clamp instead of raising an exception.
    max_size = max(1, max_size)
    overlap = max(0, min(overlap, max_size - 1))

    return _recursive_split(text, max_size, overlap, separators, metadata)
