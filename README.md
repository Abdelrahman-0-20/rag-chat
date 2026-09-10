# Document Q&A Assistant (RAG)

A retrieval-augmented generation app that answers questions about a local folder of PDFs, with citations back to the exact source file and page.

## Problem statement

Large document collections are hard to search: keyword search misses paraphrases, and dumping whole PDFs into an LLM blows the context window and costs a fortune. This project indexes a corpus of PDFs into a vector store, retrieves only the few chunks relevant to a question, and asks an LLM to answer strictly from that retrieved context. Every answer cites the chunk it came from so a human can verify it in seconds, and the model is instructed to refuse when the context does not contain the answer instead of hallucinating.

## Architecture

```
PHASE 1 - INGESTION (offline, run once)

  data/corpus/*.pdf
         |
         v
  +--------------------------------------+
  | src/ingest.py                        |
  | pypdf: extract text page by page     |
  | -> [{"text", "source", "page"}]      |
  +--------------------------------------+
         |
         v
  +--------------------------------------+
  | src/chunking.py                      |
  | recursive splitter, size + overlap   |
  | -> adds "chunk_id"                   |
  +--------------------------------------+
         |
         v
  +--------------------------------------+
  | src/embeddings.py                    |
  | SentenceTransformer (BGE small)      |
  | -> unit-normalized float32 vectors   |
  +--------------------------------------+
         |
         v
  +--------------------------------------+
  | src/vector_store.py                  |
  | faiss.IndexFlatIP + pickled metadata |
  | -> index/index.faiss                 |
  | -> index/metadata.pkl                |
  +--------------------------------------+


PHASE 2 - QUERY (online, per question)

  user question (Streamlit chat)
         |
         v
  +--------------------+     +----------------------------------+
  | src/retrieve.py    | --> | src/rerank.py (optional)         |
  | embed + FAISS      |     | CrossEncoder over top_n, keep k  |
  | top_n candidates   |     +----------------------------------+
  +--------------------+                    |
                                            v
                       +--------------------------------------+
                       | src/generate.py                      |
                       | numbered context blocks + question   |
                       | -> OpenAI chat completion            |
                       +--------------------------------------+
                                            |
                                            v
                       answer text with [1], [2] citations
                       + source chunks (file, page, score)
```

## Project structure

```
rag-doc-qa/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- config.py            # all tunables in one place
|-- build_index.py       # phase 1 CLI
|-- app.py               # phase 2 Streamlit UI
|-- src/
|   |-- __init__.py
|   |-- ingest.py        # PDF -> pages
|   |-- chunking.py      # pages -> chunks (pure Python recursive splitter)
|   |-- embeddings.py    # text -> normalized vectors
|   |-- vector_store.py  # FAISS index + metadata persistence
|   |-- retrieve.py      # question -> top-k chunks
|   |-- rerank.py        # cross-encoder reranking
|   |-- generate.py      # prompt building + LLM call + citation parsing
|   +-- evaluate.py      # recall@k over eval/qa_pairs.json
|-- eval/
|   +-- qa_pairs.json
|-- data/
|   +-- corpus/          # put your PDFs here
+-- index/               # built artifacts, git-ignored
```

## Setup

```
# 1. clone (or recreate the structure above)
git clone <your-repo-url> rag-doc-qa
cd rag-doc-qa

# 2. virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. dependencies
pip install -r requirements.txt

# 4. environment file
cp .env.example .env             # then put your real key in .env

# 5. add documents
cp /path/to/your/files.pdf data/corpus/

# 6. build the index (phase 1)
python build_index.py

# 7. run the app (phase 2)
streamlit run app.py
```

The first run of `build_index.py` downloads the embedding model (about 130 MB). If you enable reranking, the cross-encoder (about 90 MB) is downloaded on the first question.

### Running fully local

The generator talks to any OpenAI-compatible HTTP endpoint. To use Ollama:

```
ollama pull llama3.1
ollama serve
```

then in `.env`:

```
LLM_PROVIDER=local
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

The model selector in the sidebar offers `gpt-4o-mini` and `gpt-4o`; when you run a local server, set `LLM_MODEL` in `.env` to your local model name and it becomes the sidebar default.

## How it works

### Ingestion

`build_index.py` walks `data/corpus/`, opens every `.pdf` with `pypdf.PdfReader`, and emits one record per page holding the extracted text, the filename, and a 1-indexed page number. Pages with fewer than 20 characters of text are dropped, which removes blank spacers and image-only pages that would otherwise waste slots in top-k. A corrupt or password-protected PDF is logged and skipped rather than aborting the build. The page records go into a hand-written recursive character splitter that tries the coarsest separator first (`"\n\n"`), then `"\n"`, then `". "`, then `" "`, and finally falls back to a per-character split. Pieces are merged greedily up to `CHUNK_SIZE` characters, and when a piece is large enough on its own a `CHUNK_OVERLAP`-sized tail is prepended to the next piece. The overlap is what stops an answer that straddles a chunk boundary from being unretrievable. Chunks never cross a page boundary, because a citation that reads "page 12 or 13" is worthless. Each chunk gets a globally unique `chunk_id`, then the whole batch is embedded with `BAAI/bge-small-en-v1.5` and L2-normalized. Because the vectors are unit length, a FAISS `IndexFlatIP` inner product is exactly cosine similarity, which keeps the store simple: no metric conversion and no normalization at query time. The index and the parallel list of metadata dicts are written to `index/`.

`IndexFlatIP` is a brute-force flat index: exact results, no training step, no approximation. That is the right choice for a corpus up to a few hundred thousand chunks. Past that you would swap in `IndexIVFFlat` or `IndexHNSWFlat` and accept approximate recall in exchange for latency; `src/vector_store.py` is the only file that would change.

### Query

`app.py` embeds the question with the same model, searches FAISS for `TOP_N_RERANK` candidates, and optionally passes them through `cross-encoder/ms-marco-MiniLM-L-6-v2`. A bi-encoder scores the question and each chunk independently, which is fast but coarse. A cross-encoder attends over the question and chunk together, which is much more accurate but far too slow to run over the whole corpus. The standard pattern is therefore retrieve wide, rerank narrow: FAISS gives 20 cheap candidates, the cross-encoder reorders them and keeps the best 5. Reranking is a sidebar toggle so you can measure the difference yourself, and the per-chunk display shows both the cosine score and the rerank score side by side.

The surviving chunks (capped at `MAX_CONTEXT_CHUNKS`) are rendered into a prompt as numbered blocks:

```
CONTEXT:
[1]
Source: annual_report.pdf, page 12
Text: Total revenue for fiscal 2024 was 4.2 billion dollars...

[2]
Source: annual_report.pdf, page 44
Text: The independent auditor was Smith and Lane LLP...

QUESTION: what was total revenue in fiscal 2024
```

The system prompt constrains the model to answer only from those blocks, to cite with `[n]`, and to reply with the exact sentence `I don't know based on the provided documents.` when the context is insufficient. The UI parses the citations back out of the answer with a regex and reorders the sources so the chunks actually cited appear first, with filename, page, similarity score, and the chunk text. Turn on `Show LLM prompt` in the sidebar to see the exact string that was sent.

## Evaluation

Retrieval quality is measured before generation quality, because a wrong answer caused by bad retrieval cannot be fixed by a better prompt. `eval/qa_pairs.json` holds question / expected source / expected page triples; the shipped file contains ten placeholder entries for a hypothetical annual report and employee handbook, so replace the filenames, page numbers, and questions with ones from your own corpus. Page numbers are 1-indexed, matching what `src/ingest.py` records and what the UI displays.

```
python src/evaluate.py                     # top_k from config, reranking on
python src/evaluate.py --top-k 5 --no-rerank
python src/evaluate.py --top-k 10 --top-n-rerank 30
python -m src.evaluate --verbose           # equivalent, with debug logging
```

Example output from a six-chunk test corpus:

```
Index           : 6 chunks, dim 16
Embedding model : BAAI/bge-small-en-v1.5
Reranker        : disabled
Questions       : 6
top_k           : 3

  #  HIT  RANK  QUESTION                                    EXPECTED                      TOP
---------------------------------------------------------------------------------------------
  1  yes     1  what was total revenue in fiscal 2024       annual_report.pdf p.1       0.890
  2  yes     2  how many employees does the company have    annual_report.pdf p.3       0.909
  3  yes     3  who was the independent auditor             annual_report.pdf p.4       0.821
  4  yes     2  what is the remote work policy              employee_handbook.pdf p.1   0.879
  5  yes     3  how many days of annual leave               employee_handbook.pdf p.2   0.803
  6  no      -  what is the airspeed velocity of a swallow  missing.pdf p.9             0.924
---------------------------------------------------------------------------------------------
recall@3 = 5/6 = 0.833
mean rank of correct chunk (hits only) = 2.200
hits at rank 1 = 1/6
reranking = off
TOP is the cosine similarity of the highest ranked candidate.
```

recall@k is the fraction of questions where the expected source and page pair appears anywhere in the top-k retrieved chunks. Read the rank column, not just the recall number: a miss whose correct chunk sits at rank 8 means raise k or turn reranking on, while a miss where it never appears means the chunking or the embedding model is losing the signal. The script exits non-zero only when it could not run at all (missing index, empty or malformed eval file), so it is safe to use as a CI gate.

## Configuration

Every value lives in `config.py` and can be overridden with an environment variable of the same name.

| Setting | Default | Env var | Notes |
| --- | --- | --- | --- |
| `CORPUS_DIR` | `data/corpus` | `CORPUS_DIR` | folder scanned for `*.pdf`, non-recursive |
| `INDEX_DIR` | `index` | `INDEX_DIR` | where `index.faiss` and `metadata.pkl` are written |
| `EVAL_PATH` | `eval/qa_pairs.json` | `EVAL_PATH` | labelled set used by `src/evaluate.py` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | `EMBEDDING_MODEL` | 384 dims, best quality per millisecond at this size |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RERANK_MODEL` | loaded only when reranking is enabled |
| `CHUNK_SIZE` | `800` | `CHUNK_SIZE` | characters, roughly 200 tokens |
| `CHUNK_OVERLAP` | `120` | `CHUNK_OVERLAP` | characters of tail carried into the next chunk |
| `TOP_K` | `5` | `TOP_K` | chunks that end up in the prompt |
| `TOP_N_RERANK` | `20` | `TOP_N_RERANK` | candidates FAISS returns before reranking |
| `MAX_CONTEXT_CHUNKS` | `8` | `MAX_CONTEXT_CHUNKS` | hard cap on prompt blocks, bounds cost |
| `EMBED_BATCH_SIZE` | `64` | `EMBED_BATCH_SIZE` | texts per forward pass while building the index |
| `LLM_PROVIDER` | `openai` | `LLM_PROVIDER` | `openai`, or `local` for a compatible server |
| `LLM_MODEL` | `gpt-4o-mini` | `LLM_MODEL` | chat model name and sidebar default |
| `LLM_TEMPERATURE` | `0.0` | `LLM_TEMPERATURE` | deterministic answers for grounded QA |
| `OPENAI_API_KEY` | empty | `OPENAI_API_KEY` | required when provider is `openai` |
| `OPENAI_BASE_URL` | empty | `OPENAI_BASE_URL` | set for Ollama, vLLM, LM Studio, LiteLLM |

`build_index.py` also accepts `--corpus-dir`, `--index-dir`, `--chunk-size`, `--chunk-overlap`, and `--embedding-model` so you can rebuild and re-measure recall without editing any file. Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL` requires a rebuild; changing `TOP_K`, `TOP_N_RERANK`, or `LLM_MODEL` does not.

## What I would do next

- Hybrid search: BM25 over the same chunks, fused with the dense scores using reciprocal rank fusion. Dense retrieval misses exact identifiers (part numbers, clause names, error codes) that lexical search finds instantly.
- Parent-child retrieval: index small chunks for precise matching but return the surrounding page section to the LLM, so the answer has enough local context without inflating the prompt with irrelevant chunks.
- Query rewriting: one cheap LLM call to expand the question into two or three search-friendly variants, plus a HyDE-style hypothetical answer embedding. Fixes the common failure where a conversational question embeds poorly.
- Multi-hop QA: an agent loop that retrieves, checks whether the answer is complete, and issues follow-up retrievals. Needed for questions such as "compare the 2023 and 2024 margins" whose evidence sits far apart in similarity space.
- Caching: a persistent cache keyed on (question, index fingerprint) for embeddings and answers, plus an LRU for repeated prompts. Most real query traffic is duplicated.
- Observability: log every retrieval with its scores, the prompt, the answer, and latency to a trace store, then add thumbs-up and thumbs-down buttons in the UI to build a labelled evaluation set from real traffic instead of the hand-written `qa_pairs.json`.
- Incremental indexing: hash each PDF and only re-embed files that changed, so adding one document to a large corpus does not cost a full rebuild.
- OCR: run `ocrmypdf` over scanned documents during ingestion so image-only PDFs become searchable instead of silently contributing nothing.
- Generation-level evaluation: faithfulness and answer-relevance scoring, once the retrieval numbers are solid enough that a wrong answer is the model's fault rather than the retriever's.