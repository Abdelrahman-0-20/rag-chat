"""Load PDF files into page-level records."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def count_pdfs(corpus_dir: Path | str) -> int:
    return len(list(Path(corpus_dir).glob("*.pdf")))


def load_pdfs(corpus_dir: Path | str) -> list[dict]:
    corpus_dir = Path(corpus_dir)
    pages: list[dict] = []
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            print(f"  Skipping {pdf_path.name}: {exc}")
            continue
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
            text = text.strip()
            if text:
                pages.append({"source": pdf_path.name, "page": page_number, "text": text})
    return pages