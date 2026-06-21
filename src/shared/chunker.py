"""Découpage récursif de texte avec chevauchement pour les systèmes RAG.

Découpe un texte selon une hiérarchie de séparateurs (paragraphe, ligne,
phrase, puis mot) et applique un chevauchement de caractères configurable afin
de préserver le contexte aux frontières des chunks.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Un morceau de texte avec ses métadonnées et son index de position."""

    text: str
    metadata: dict = field(default_factory=dict)
    index: int = 0


_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _make_chunk(text: str, index: int, metadata: dict | None) -> Chunk:
    """Construit un Chunk avec une copie des métadonnées et la stratégie."""
    meta = dict(metadata) if metadata else {}
    meta["chunk_strategy"] = "recursive"
    return Chunk(text=text, metadata=meta, index=index)


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Découpe `text` sur `separator`, en ignorant les fragments vides."""
    if separator == "":
        return list(text)  # dernier recours : caractère par caractère
    return [part for part in text.split(separator) if part.strip()]


def _split_into_windows(text: str, size: int, step: int) -> list[str]:
    """Découpe `text` en fenêtres de `size` caractères avançant de `step`."""
    return [
        window
        for start in range(0, len(text), step)
        if (window := text[start:start + size].strip())
    ]


# ---------------------------------------------------------------------------
# Étapes du découpage récursif
# ---------------------------------------------------------------------------

def _normalize_segments(
    segments: list[str],
    size: int, 
    remaining_separators: list[str],
) -> list[str]:
    """Garantit que chaque segment tient dans `size`.

    Un segment trop grand est redécoupé avec les séparateurs restants, ou à
    défaut en fenêtres de caractères. Le chevauchement n'est pas appliqué ici :
    il l'est à l'assemblage, sur la liste finale des segments.
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
    """Regroupe les segments en chunks d'au plus `size` caractères.

    Les `overlap` derniers caractères d'un chunk sont répétés au début du
    suivant pour préserver le contexte.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for seg in segments:
        added = (len(join_sep) if current else 0) + len(seg)
        if current and current_len + added > size:
            # Le segment déborde : on clôt le chunk courant, puis on amorce le
            # suivant avec la fin du précédent (le chevauchement).
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
    """Choisit le premier séparateur qui découpe, normalise, puis assemble."""
    if len(text) <= size:
        stripped = text.strip()
        return [_make_chunk(stripped, 0, metadata)] if stripped else []

    for i, sep in enumerate(separators):
        segments = _split_by_separator(text, sep)
        if len(segments) <= 1:
            continue  # ce séparateur ne découpe rien : on passe au suivant
        fine = _normalize_segments(segments, size, separators[i + 1:])
        return _build_chunks_with_overlap(fine, size, overlap, sep, metadata)

    # Aucun séparateur n'a fonctionné : fenêtres de caractères chevauchantes.
    # `step` est borné à 1 pour ne jamais boucler indéfiniment ni perdre de texte.
    windows = _split_into_windows(text, size, max(size - overlap, 1))
    return [_make_chunk(window, idx, metadata) for idx, window in enumerate(windows)]


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def recursive_chunk(
    text: str,
    max_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Découpe `text` récursivement selon une hiérarchie de séparateurs.

    Essaie de couper par paragraphe, puis ligne, puis phrase, puis mot. Chaque
    chunk fait au plus `max_size` caractères (hors chevauchement repris du chunk
    précédent), et `overlap` caractères de fin sont répétés au début du suivant.
    `metadata` est copié sur chaque chunk renvoyé.
    """
    if separators is None:
        separators = _DEFAULT_SEPARATORS

    # Normalisation défensive : un appelant peut fournir overlap >= max_size ou
    # un max_size <= 0, ce qui rendrait le pas de découpage <= 0 (plantage ou
    # perte de texte). On borne au lieu de lever une exception.
    max_size = max(1, max_size)
    overlap = max(0, min(overlap, max_size - 1))

    return _recursive_split(text, max_size, overlap, separators, metadata)
