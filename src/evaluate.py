

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python src/evaluate.py` as well as `python -m src.evaluate` by putting
# the project root on sys.path when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.rerank import Reranker  # noqa: E402
from src.retrieve import retrieve  # noqa: E402
from src.vector_store import VectorStore, index_exists  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """One labelled evaluation example."""

    question: str
    expected_source: str
    expected_page: int


@dataclass
class EvalResult:
    """Outcome of evaluating one question."""

    question: str
    expected_source: str
    expected_page: int
    hit: bool
    rank: int  # 1-indexed position of the first matching chunk, 0 if not found
    top_score: float


def load_qa_pairs(path: Path) -> list[QAPair]:
    """Read the evaluation set, skipping malformed entries with a warning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"evaluation file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of objects")

    pairs: list[QAPair] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning("entry %d is not an object, skipping", index)
            continue
        question = str(item.get("question", "")).strip()
        source = str(item.get("expected_source", "")).strip()
        page = item.get("expected_page")
        if not question or not source or page is None:
            logger.warning(
                "entry %d is missing question/expected_source/expected_page, skipping",
                index,
            )
            continue
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            logger.warning("entry %d has a non-integer expected_page, skipping", index)
            continue
        pairs.append(QAPair(question=question, expected_source=source, expected_page=page_number))

    return pairs


def _matches(chunk: dict, pair: QAPair) -> bool:
    """Return True when a retrieved chunk is the expected source and page."""
    chunk_source = Path(str(chunk.get("source", ""))).name
    expected_source = Path(pair.expected_source).name
    try:
        chunk_page = int(chunk.get("page", -1))
    except (TypeError, ValueError):
        return False
    return chunk_source == expected_source and chunk_page == pair.expected_page


def evaluate_pair(
    pair: QAPair,
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker | None,
    top_k: int,
    top_n_rerank: int,
) -> EvalResult:
    """Run retrieval (and optional reranking) for one question and score it.

    The candidate pool is `max(top_k, top_n_rerank)` so that, with reranking
    off, the reported rank can still exceed k and tell you how far away the
    correct chunk was rather than just that it missed.
    """
    pool_size = max(top_k, top_n_rerank) if reranker is not None else top_k * 4
    candidates = retrieve(pair.question, store, embedder, pool_size)

    if reranker is not None:
        candidates = reranker.rerank(pair.question, candidates, top_k)

    rank = 0
    for position, chunk in enumerate(candidates[:top_k], start=1):
        if _matches(chunk, pair):
            rank = position
            break

    return EvalResult(
        question=pair.question,
        expected_source=pair.expected_source,
        expected_page=pair.expected_page,
        hit=rank > 0,
        rank=rank,
        top_score=candidates[0]["score"] if candidates else 0.0,
    )


def _truncate(text: str, width: int) -> str:
    """Truncate `text` to `width` columns for table printing."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def print_report(results: list[EvalResult], top_k: int, reranking: bool) -> float:
    """Print the per-question table and the overall recall@k, and return recall."""
    total = len(results)
    hits = sum(1 for result in results if result.hit)
    recall = hits / total if total else 0.0

    question_width = max(
        [len("QUESTION")] + [min(len(" ".join(r.question.split())), 52) for r in results]
    )
    source_width = max(
        [len("EXPECTED")]
        + [len(f"{r.expected_source} p.{r.expected_page}") for r in results]
    )

    header = (
        f"{'#':>3}  {'HIT':<4} {'RANK':>4}  "
        f"{'QUESTION':<{question_width}}  {'EXPECTED':<{source_width}}  {'TOP':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for index, result in enumerate(results, start=1):
        expected = f"{result.expected_source} p.{result.expected_page}"
        rank_cell = str(result.rank) if result.hit else "-"
        print(
            f"{index:>3}  {'yes' if result.hit else 'no':<4} {rank_cell:>4}  "
            f"{_truncate(result.question, question_width):<{question_width}}  "
            f"{expected:<{source_width}}  {result.top_score:>6.3f}"
        )
    print("-" * len(header))
    print(f"recall@{top_k} = {hits}/{total} = {recall:.3f}")

    hit_ranks = [result.rank for result in results if result.hit]
    if hit_ranks:
        mean_rank = sum(hit_ranks) / len(hit_ranks)
        first_rank_hits = sum(1 for rank in hit_ranks if rank == 1)
        print(f"mean rank of correct chunk (hits only) = {mean_rank:.3f}")
        print(f"hits at rank 1 = {first_rank_hits}/{total}")
    print(f"reranking = {'on' if reranking else 'off'}")
    print("TOP is the cosine similarity of the highest ranked candidate.")
    print()
    return recall


def run(
    qa_path: Path,
    index_dir: Path,
    top_k: int,
    use_reranker: bool,
    top_n_rerank: int,
    embedding_model: str,
    rerank_model: str,
) -> float | None:
    """Load the index and evaluation set, score every question, print the report."""
    index_dir = Path(index_dir)
    if not index_exists(index_dir):
        print(f"No index found in {index_dir}.")
        print("Build one first:  python build_index.py")
        return None

    pairs = load_qa_pairs(qa_path)
    if not pairs:
        print(f"No usable entries in {qa_path}.")
        return None

    store = VectorStore.load(index_dir)
    embedder = Embedder(embedding_model)
    reranker = Reranker(rerank_model) if use_reranker else None

    print(f"Index           : {store.stats()['vectors']} chunks, dim {store.dimension}")
    print(f"Embedding model : {embedding_model}")
    print(f"Reranker        : {rerank_model if use_reranker else 'disabled'}")
    print(f"Questions       : {len(pairs)}")
    print(f"top_k           : {top_k}")
    if use_reranker:
        print(f"top_n_rerank    : {top_n_rerank}")

    results = [
        evaluate_pair(pair, store, embedder, reranker, top_k, top_n_rerank)
        for pair in pairs
    ]
    return print_report(results, top_k, use_reranker)


def main() -> int:
    """Parse CLI arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Measure retrieval recall@k over eval/qa_pairs.json"
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=CONFIG.resolved_eval_path(),
        help="path to the JSON evaluation set (default: eval/qa_pairs.json)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=CONFIG.resolved_index_dir(),
        help="directory holding index.faiss and metadata.pkl (default: index/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=CONFIG.TOP_K,
        help=f"how many chunks count as a hit (default: {CONFIG.TOP_K})",
    )
    parser.add_argument(
        "--top-n-rerank",
        type=int,
        default=CONFIG.TOP_N_RERANK,
        help=f"candidate pool passed to the reranker (default: {CONFIG.TOP_N_RERANK})",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="evaluate cosine retrieval only, without the cross-encoder",
    )
    parser.add_argument(
        "--embedding-model",
        default=CONFIG.EMBEDDING_MODEL,
        help="must match the model the index was built with",
    )
    parser.add_argument(
        "--rerank-model",
        default=CONFIG.RERANK_MODEL,
        help="cross-encoder id used when reranking is enabled",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        recall = run(
            qa_path=args.qa_file,
            index_dir=args.index_dir,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
            top_n_rerank=args.top_n_rerank,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1

    # None means the evaluation could not run at all (missing index, empty or
    # malformed eval file); a real 0.000 recall is still a valid result.
    return 1 if recall is None else 0


if __name__ == "__main__":
    raise SystemExit(main())