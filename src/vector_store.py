

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.pkl"


def index_exists(index_dir: Path) -> bool:
    """Return True when both the FAISS index and its metadata are on disk."""
    index_dir = Path(index_dir)
    return (index_dir / INDEX_FILENAME).exists() and (
        index_dir / METADATA_FILENAME
    ).exists()


class VectorStore:
    """Thin wrapper around `faiss.IndexFlatIP` plus a parallel metadata list."""

    def __init__(self, dimension: int, metadatas: list[dict] | None = None) -> None:
        """Create an empty store for vectors of `dimension` floats."""
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadatas: list[dict] = list(metadatas) if metadatas else []

    def __len__(self) -> int:
        """Return the number of vectors currently in the index."""
        return int(self.index.ntotal)

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        """Append vectors and their metadata, keeping the two lists aligned.

        Args:
            embeddings: (n, dimension) float32 array, ideally unit-normalized.
            metadatas: n dicts, one per row, in the same order.
        """
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (n, dim), got shape {vectors.shape}"
            )
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dim {vectors.shape[1]} does not match "
                f"index dim {self.dimension}"
            )
        if vectors.shape[0] != len(metadatas):
            raise ValueError(
                f"got {vectors.shape[0]} vectors but {len(metadatas)} metadata rows"
            )
        if vectors.shape[0] == 0:
            logger.warning("VectorStore.add called with zero rows, nothing to do")
            return

        self.index.add(vectors)
        self.metadatas.extend(metadatas)
        logger.info("added %d vectors (total %d)", vectors.shape[0], len(self))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Return the `top_k` most similar chunks, highest score first.

        Each result dict carries the stored metadata plus a `score` field,
        which is cosine similarity in [-1, 1] given normalized embeddings.
        FAISS returns -1 for slots it could not fill (when the index holds
        fewer vectors than top_k); those are filtered out.
        """
        if len(self) == 0:
            return []
        if top_k <= 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query dim {query.shape[1]} does not match index dim {self.dimension}"
            )

        effective_k = min(int(top_k), len(self))
        scores, ids = self.index.search(query, effective_k)

        results: list[dict] = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0 or row_id >= len(self.metadatas):
                continue
            record: dict[str, Any] = dict(self.metadatas[row_id])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self, path: Path) -> None:
        """Write the FAISS index and the metadata list into directory `path`."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        with open(directory / METADATA_FILENAME, "wb") as handle:
            pickle.dump(self.metadatas, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "saved index (%d vectors, dim %d) to %s",
            len(self),
            self.dimension,
            directory,
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a store previously written by `save`.

        Raises:
            FileNotFoundError: if either of the two files is missing, which
                usually means `python build_index.py` has not been run yet.
            ValueError: if the index and metadata disagree on row count, which
                means one of the two files is stale.
        """
        directory = Path(path)
        index_file = directory / INDEX_FILENAME
        metadata_file = directory / METADATA_FILENAME

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"no index found in {directory}. Run: python build_index.py"
            )

        index = faiss.read_index(str(index_file))
        with open(metadata_file, "rb") as handle:
            metadatas = pickle.load(handle)

        # VALIDATION: verify the deserialized metadata is a well-formed list
        # of dicts with the keys downstream code expects.  A corrupted or
        # tampered pkl file would otherwise crash at query time with an
        # unhelpful KeyError or, worse, execute arbitrary code via pickle.
        if not isinstance(metadatas, list):
            raise ValueError(
                f"metadata in {metadata_file} is not a list "
                f"(got {type(metadatas).__name__}); rebuild the index"
            )
        _REQUIRED_KEYS = {"text", "source", "page"}
        for i, entry in enumerate(metadatas[:10]):  # check first 10 for perf
            if not isinstance(entry, dict):
                raise ValueError(
                    f"metadata[{i}] is {type(entry).__name__}, expected dict; "
                    f"rebuild the index"
                )
            missing = _REQUIRED_KEYS - entry.keys()
            if missing:
                raise ValueError(
                    f"metadata[{i}] is missing keys {missing}; rebuild the index"
                )

        if index.ntotal != len(metadatas):
            raise ValueError(
                f"index holds {index.ntotal} vectors but metadata holds "
                f"{len(metadatas)} rows; rebuild the index"
            )

        store = cls(dimension=index.d, metadatas=metadatas)
        store.index = index
        logger.info("loaded index with %d vectors from %s", len(store), directory)
        return store

    def stats(self) -> dict[str, Any]:
        """Return a small dict describing the store, for logging and the UI."""
        sources = {meta.get("source") for meta in self.metadatas}
        return {
            "vectors": len(self),
            "dimension": self.dimension,
            "sources": len(sources),
        }

    def __repr__(self) -> str:
        """Return a debug representation with the vector count and dimension."""
        return f"VectorStore(vectors={len(self)}, dimension={self.dimension})"