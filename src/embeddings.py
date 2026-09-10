"""Embedding wrapper around sentence-transformers.

The class exists so that the rest of the codebase never touches the model
directly. Swapping BGE for another sentence encoder, or for an API-based
embedder, means changing this one file.

Design decisions:
  * Lazy loading. Importing this module costs nothing; the ~130 MB model is
    downloaded and loaded on the first embed call. That keeps `build_index.py`
    and the Streamlit import phase fast and lets unit tests import the module
    without a model on disk.
  * L2 normalization on the way out. With unit-length vectors, a dot product is
    exactly cosine similarity, so the FAISS IndexFlatIP scores are directly
    interpretable in [-1, 1] and nothing downstream needs a metric conversion.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy-loaded sentence encoder producing unit-norm float32 vectors."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        """Store the model name; the model itself is loaded on first use.

        Args:
            model_name: Hugging Face sentence-transformers model id.
            batch_size: texts per forward pass during bulk embedding.
            device: "cpu", "cuda", or None to let the library choose.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """Load and cache the SentenceTransformer on first access."""
        if self._model is None:
            # Imported here, not at module level: sentence-transformers pulls in
            # torch, which is hundreds of megabytes and seconds of import time.
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "embedding model ready (dim=%d)", self._model.get_sentence_embedding_dimension()
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality of the underlying model."""
        return int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows to unit length, leaving zero rows untouched."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            norm = float(np.linalg.norm(vectors))
            return vectors / norm if norm > 0 else vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against dividing by zero for an empty or all-zero input row.
        safe = np.where(norms == 0, 1.0, norms)
        return (vectors / safe).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a normalized (n, dim) float32 array."""
        if not texts:
            # Return a correctly shaped empty array so callers can concatenate
            # and so VectorStore.add never sees a ragged input.
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalized below, explicitly
            show_progress_bar=False,
        )
        return self._normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a normalized (1, dim) float32 array.

        Returns a 2-D row vector because that is what faiss.search expects;
        passing a 1-D array raises inside FAISS with an unhelpful message.
        """
        return self.embed_texts([text])

    def __repr__(self) -> str:
        """Return a debug representation showing whether the model is loaded."""
        state = "loaded" if self._model is not None else "not loaded"
        return f"Embedder(model_name={self.model_name!r}, {state})"