# rag-chat

**Ask questions about your PDFs. Get answers grounded in the source, with
citations back to the file and page.**

A Streamlit application that indexes a folder of PDF documents into a
vector store, retrieves the chunks most relevant to a question, and asks
an LLM to answer using **only** those chunks. Every answer cites the
source, and the model is instructed to say "I don't know" rather than
guess when the context doesn't contain the answer.


---

## Requirements

Before you start, make sure you have:

| Requirement | Why | How to get it |
|---|---|---|
| **Python 3.10+** | Runtime | [python.org](https://www.python.org/downloads/) |
| **OpenRouter API key** | Used to call the LLM that generates answers | [openrouter.ai/keys](https://openrouter.ai/keys) — free tier available |
| **~500 MB free disk** | For the two model downloads | — |
| **At least one PDF** | The documents you want to query | Yours |

### ⚠️ You must have an OpenRouter API key

This project **will not generate answers without an API key**. Retrieval
will still work — you'll see the chunks that matched your question — but
the answer text will be empty.

1. Create a free account at [openrouter.ai](https://openrouter.ai)
2. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
3. Click **Create Key** and copy it (it starts with `sk-or-v1-`)
4. Paste it into your `.env` file as shown in the setup below

OpenRouter offers free models (look for names ending in `:free`) and paid
models. The default in this project is `openai/gpt-4o-mini`, which costs
fractions of a cent per question.

> **Never commit your `.env` file.** It contains your key and is already
> listed in `.gitignore`. If you ever push it by accident, revoke the key
> immediately at [openrouter.ai/keys](https://openrouter.ai/keys) and
> create a new one.

---

## Features

- **Two-stage retrieval** — fast bi-encoder search, then an optional
  cross-encoder rerank. The reranker is a sidebar toggle so you can see
  what it changes.
- **Source-level traceability** — every chunk carries its original filename
  and 1-indexed page number. Preserved through chunking, displayed in the UI.
- **Configurable at runtime** — model, top-k, reranking, and prompt
  visibility are all sidebar controls. No code edits required.
- **Provider-agnostic** — OpenRouter by default, but any OpenAI-compatible
  endpoint works (Ollama, LM Studio, vLLM).
- **Rebuild without editing code** — `build_index.py` accepts CLI flags
  for corpus directory, index directory, chunk size, overlap, and embedding
  model.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Abdelrahman-0-20/rag-chat.git
cd rag-chat

# 2. Virtual environment
python -m venv venv
# Windows:       venv\Scripts\activate
# macOS / Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
#    Open .env in a text editor and set OPENROUTER_API_KEY to your key.
#    Get one at https://openrouter.ai/keys if you don't have one.

# 5. Add your documents
cp /path/to/*.pdf data/corpus/

# 6. Build the search index
python build_index.py

# 7. Run the app
streamlit run app.py


INGESTION  (offline, run once)

  data/corpus/*.pdf
        │
        ▼
  src/ingest.py        ──►  [{source, page, text}, ...]
        │
        ▼
  src/chunking.py      ──►  [{chunk_id, source, page, text}, ...]
        │
        ▼
  src/embeddings.py    ──►  float32 (N, 384), L2-normalized
        │
        ▼
  src/vector_store.py  ──►  index/faiss.index
                            index/metadata.json


QUERY  (per question)

  user question
        │
        ▼
  src/retrieve.py      ──►  top-N candidates   (FAISS + bi-encoder)
        │
        ▼
  src/rerank.py        ──►  top-K kept         (cross-encoder, optional)
        │
        ▼
  src/generate.py      ──►  answer with [n] citations  (OpenRouter)
        │
        ▼
  app.py               ──►  renders answer + Sources expander
