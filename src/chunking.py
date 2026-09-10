
from __future__ import annotations

# Ordered coarsest to finest. The trailing empty string is the fallback that
# splits per character, which guarantees termination even for a long run with
# no whitespace or punctuation at all.
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def _split_once(text: str, separator: str) -> list[str]:
    """Split `text` on `separator`, keeping the separator attached to the left piece."""
    if separator == "":
        return list(text)

    parts = text.split(separator)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            pieces.append(part)
        else:
            pieces.append(part + separator)
    return pieces


def _find_separator(text: str, separators: list[str]) -> tuple[str, list[str]]:
    """Return the first separator present in `text` and the finer ones after it."""
    for index, candidate in enumerate(separators):
        if candidate == "":
            return "", [""]
        if candidate in text:
            return candidate, separators[index + 1:]
    # Nothing matched: fall back to the per-character split.
    return "", [""]


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split `text` into pieces no longer than `chunk_size`, respecting separators."""
    if not text.strip():
        return []

    # Base case: already small enough.
    if len(text) <= chunk_size:
        return [text]

    # Degenerate configuration: a one-character budget cannot be honoured by a
    # sentence- or word-aware splitter, so return the block whole rather than
    # recursing forever.
    if chunk_size <= 1:
        return [text]

    separator, remaining = _find_separator(text, separators)
    raw_pieces = _split_once(text, separator)

    # A single piece means the chosen separator did not cut anything. Drop to
    # the per-character split so the recursion always makes progress.
    if len(raw_pieces) <= 1:
        return _split_recursive(text, [""], chunk_size)

    pieces: list[str] = []
    for piece in raw_pieces:
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            # Recurse with the finer separators only. `remaining` is always
            # strictly finer than `separators`, so this terminates.
            pieces.extend(_split_recursive(piece, remaining, chunk_size))

    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks and carry the overlap forward.

    The size check happens before a piece is appended, never after. Checking
    after (the naive formulation) lets the buffer sit above `chunk_size` until
    the next piece arrives, and if no next piece arrives the final flush emits
    an oversized chunk. Checking first makes `sum(len(p) for p in buffer) <=
    chunk_size` an invariant of the loop.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for piece in pieces:
        if buffer_len + len(piece) > chunk_size:
            merged = "".join(buffer).strip()
            if merged:
                chunks.append(merged)

            # Carry the tail of the chunk just emitted into the next one, so a
            # fact straddling the boundary is still retrievable from one side.
            # The window is taken from the joined string rather than the piece
            # list, which makes it a true character window across sentences.
            overlap = merged[-chunk_overlap:].lstrip() if chunk_overlap > 0 else ""

            if overlap and len(overlap) + len(piece) <= chunk_size:
                buffer, buffer_len = [overlap], len(overlap)
            else:
                # Rare path: an indivisible piece nearly as long as the whole
                # chunk. Honouring the overlap here would either exceed the
                # size limit or emit a stub chunk made of duplicated text, so
                # drop the overlap for this one boundary instead.
                buffer, buffer_len = [], 0

        buffer.append(piece)
        buffer_len += len(piece)

    tail = "".join(buffer).strip()
    if tail:
        chunks.append(tail)

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a single string into overlapping chunks of at most `chunk_size` chars."""
    if not text or not text.strip():
        return []
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        # An overlap at or above the chunk size would make every chunk a prefix
        # of the next one and never advance. Fail loudly at build time.
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    pieces = _split_recursive(text, SEPARATORS, chunk_size)
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Chunk page dicts into metadata-annotated chunks with global ids.

    Args:
        pages: list of {"text": str, "source": str, "page": int}.
        chunk_size: maximum chunk length in characters.
        chunk_overlap: characters of the previous chunk repeated in the next.

    Returns:
        list of {"text": str, "source": str, "page": int, "chunk_id": int}.
        `chunk_id` is unique across the whole corpus and `page` is carried
        through unchanged (1-indexed, as produced by src/ingest.py).

    Chunks never cross a page boundary. Crossing pages would make the `page`
    citation ambiguous, and citation accuracy is the point of the app. The cost
    is a few short chunks at page ends, which the overlap largely absorbs.
    """
    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        for chunk_text in split_text(
            page.get("text", ""), chunk_size, chunk_overlap
        ):
            chunks.append(
                {
                    "text": chunk_text,
                    "source": page.get("source", ""),
                    "page": page.get("page", 0),
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1

    return chunks