
from __future__ import annotations

import logging

from src.embeddings import Embedder
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int,
) -> list[dict]:
    """Embed `question` and return the `top_k` most similar chunks.

    Args:
        question: natural language query from the user.
        store: loaded VectorStore to search.
        embedder: Embedder using the same model the index was built with.
        top_k: how many chunks to return. When reranking is enabled downstream,
            callers pass TOP_N_RERANK here (a wide, cheap candidate pool) and
            let src/rerank.py cut it down to the real top_k.

    Returns:
        list of dicts, each with text, source, page, chunk_id and score
        (cosine similarity in [-1, 1], descending). Empty list if the index is
        empty or top_k is not positive.
    """
    question = (question or "").strip()
    if not question:
        logger.warning("retrieve called with an empty question")
        return []
    if top_k <= 0:
        return []
    if len(store) == 0:
        logger.warning("retrieve called against an empty index")
        return []

    query_vector = embedder.embed_query(question)
    results = store.search(query_vector, top_k)

    logger.info(
        "retrieved %d chunk(s) for question of %d char(s), best score %.4f",
        len(results),
        len(question),
        results[0]["score"] if results else 0.0,
    )
    return results


def format_context_block(index: int, chunk: dict) -> str:
    """Render one retrieved chunk as a numbered, citable context block.

    Shared by src/generate.py (prompt building) and app.py (UI display) so the
    [n] numbering a user sees in the answer always matches the [n] numbering
    shown in the sources panel.
    """
    return (
        f"[{index}]\n"
        f"Source: {chunk.get('source', 'unknown')}, "
        f"page {chunk.get('page', '?')}\n"
        f"Text: {chunk.get('text', '').strip()}"
    )