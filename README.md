# rag-chat

A Streamlit app for question-answering over a folder of PDFs. Answers are
grounded in retrieved chunks and cite the source file and page, so you can
verify every claim in seconds.

<!-- Add a screenshot here. One image does more than the next 500 words.
     ![Screenshot](docs/screenshot.png)
-->

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.40%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

- Indexes every PDF in `data/corpus/` into a FAISS vector store.
- Retrieves the chunks most relevant to a question, optionally reranking
  them with a cross-encoder.
- Asks an LLM (via OpenRouter by default) to answer **only** from those
  chunks and to cite them as `[1]`, `[2]`.
- Displays the answer alongside the chunks that were cited, with the
  similarity score and rerank score for each.

If the retrieved context doesn't contain the answer, the model says so
instead of guessing.

## Features

- **Two-stage retrieval** — fast bi-encoder search followed by an optional
  cross-encoder rerank. The reranker is a sidebar toggle so you can see
  what it changes.
- **Source-level traceability** — every chunk carries its original filename
  and 1-indexed page number, preserved through chunking and displayed in
  the UI.
- **Configurable in the UI** — model, top-k, reranking, top-n-rerank, and
  whether to show sources or the raw prompt are all sidebar controls.
- **Local or remote LLM** — OpenRouter by default, but any
  OpenAI-compatible endpoint works (Ollama, LM Studio, vLLM) by pointing
  `OPENROUTER_BASE_URL` at it.
- **Rebuild without editing code** — `build_index.py` accepts CLI flags
  for corpus directory, index directory, chunk size, chunk overlap, and
  embedding model.

## Quick start

Requires Python 3.10 or newer and roughly 500 MB of free disk space for
model weights.

```bash
# 1. clone
git clone https://github.com/Abdelrahman-0-20/rag-chat.git
cd rag-chat

# 2. virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 3. dependencies
pip install -r requirements.txt

# 4. environment
cp .env.example .env
# open .env and set OPENROUTER_API_KEY to your key
# get one at https://openrouter.ai/keys

# 5. put your PDFs in the corpus folder
cp /path/to/*.pdf data/corpus/

# 6. build the index
python build_index.py

# 7. run the app
streamlit run app.py
