

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Lazy-loaded CrossEncoder wrapper that rescores (question, chunk) pairs."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """Store the model name; the model loads on the first rerank call.

        Args:
            model_name: Hugging Face cross-encoder id, e.g.
                "cross-encoder/ms-marco-MiniLM-L-6-v2".
            device: "cpu", "cuda", or None to let the library choose.
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Load and cache the CrossEncoder on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading reranker %s", self.model_name)
            # No activation function is forced: the checkpoint's own config
            # decides (sigmoid for the ms-marco models, which puts the score in
            # [0, 1]). Ranking order is unaffected by any monotonic activation,
            # so this is purely about how interpretable the displayed number is.
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rescore `candidates` against `question` and keep the best `top_k`.

        Args:
            question: the user query.
            candidates: chunk dicts as returned by src.retrieve.retrieve.
            top_k: how many chunks to keep after reranking.

        Returns:
            New list of chunk dicts sorted by descending `rerank_score`. The
            original `score` (cosine similarity) is preserved so the UI can
            show both. Input dicts are copied, never mutated.

        Only the raw chunk text is scored, not the numbered prompt block. The
        cross-encoder was trained on plain (query, passage) pairs, and feeding
        it "[3]\\nSource: report.pdf, page 7" adds tokens it never saw in
        training plus a positional hint that can bias the ordering.
        """
        if not candidates:
            return []
        if top_k <= 0:
            return []
        question = (question or "").strip()
        if not question:
            return list(candidates)[:top_k]

        pairs = [[question, str(chunk.get("text", ""))] for chunk in candidates]
        scores = self.model.predict(pairs)

        scored: list[dict] = []
        for chunk, rerank_score in zip(candidates, scores):
            record = dict(chunk)
            record["rerank_score"] = float(rerank_score)
            scored.append(record)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        kept = scored[:top_k]

        logger.info(
            "reranked %d candidates down to %d (best %.4f, worst kept %.4f)",
            len(candidates),
            len(kept),
            kept[0]["rerank_score"] if kept else 0.0,
            kept[-1]["rerank_score"] if kept else 0.0,
        )
        return kept

    def __repr__(self) -> str:
        """Return a debug representation showing whether the model is loaded."""
        state = "loaded" if self._model is not None else "not loaded"
        return f"Reranker(model_name={self.model_name!r}, {state})"