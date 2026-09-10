from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from src.chunking import chunk_pages
from src.embeddings import Embedder
from src.ingest import count_pdfs, load_pdfs
from src.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the retrieval index.")
    p.add_argument("--corpus-dir", type=Path, default=CONFIG.resolved_corpus_dir())
    p.add_argument("--index-dir", type=Path, default=CONFIG.resolved_index_dir())
    p.add_argument("--chunk-size", type=int, default=CONFIG.CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=CONFIG.CHUNK_OVERLAP)
    p.add_argument("--embedding-model", default=CONFIG.EMBEDDING_MODEL)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _human_size(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024 * 1024: return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    corpus_dir: Path = args.corpus_dir
    index_dir: Path = args.index_dir

    if args.chunk_overlap >= args.chunk_size:
        print("error: --chunk-overlap must be smaller than --chunk-size")
        return 1

    started = time.perf_counter()
    print(f"Corpus directory   : {corpus_dir}")
    print(f"Index directory    : {index_dir}")
    print(f"Embedding model    : {args.embedding_model}")
    print(f"Chunk size/overlap : {args.chunk_size} / {args.chunk_overlap}")
    print()

    pdf_count = count_pdfs(corpus_dir)
    if pdf_count == 0:
        print(f"No PDFs found in {corpus_dir}")
        return 1

    print(f"Reading {pdf_count} PDF(s)...")
    pages = load_pdfs(corpus_dir)
    if not pages:
        print("No text extracted.")
        return 1
    print(f"  extracted {len(pages)} page(s)")

    print("Chunking...")
    chunks = chunk_pages(pages, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("error: no chunks produced")
        return 1
    lengths = [len(c["text"]) for c in chunks]
    print(f"  {len(chunks)} chunk(s), min/mean/max = "
          f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

    print("Embedding (first run downloads the model)...")
    embedder = Embedder(args.embedding_model, batch_size=CONFIG.EMBED_BATCH_SIZE)
    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    print(f"  embedded {embeddings.shape[0]} chunks into {embeddings.shape[1]} dims")

    print("Building FAISS index...")
    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save(index_dir)

    elapsed = time.perf_counter() - started
    index_bytes = sum(p.stat().st_size for p in index_dir.iterdir() if p.is_file())
    print()
    print("=" * 56)
    print("Index build summary")
    print("=" * 56)
    print(f"PDFs found         : {pdf_count}")
    print(f"Pages with text    : {len(pages)}")
    print(f"Chunks indexed     : {len(chunks)}")
    print(f"Index dimension    : {store.dimension}")
    print(f"Index size on disk : {_human_size(index_bytes)}")
    print(f"Index directory    : {index_dir}")
    print(f"Build time         : {elapsed:.2f} seconds")
    print("=" * 56)
    print()
    print("Next step:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())