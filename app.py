"""Streamlit Q&A UI. Python-3.14-compatible: no st.spinner, no @st.cache_resource."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from src.embeddings import Embedder
from src.generate import (
    build_prompt, cited_contexts, generate_answer, parse_citations,
)
from src.rerank import Reranker
from src.retrieve import retrieve
from src.vector_store import VectorStore, index_exists

_LLM_MODELS_BASE = ["gpt-4o-mini", "gpt-4o"]
LLM_MODELS = (
    [CONFIG.LLM_MODEL] + _LLM_MODELS_BASE
    if CONFIG.LLM_MODEL not in _LLM_MODELS_BASE
    else list(_LLM_MODELS_BASE)
)

_CACHE: dict[str, Any] = {}


def get_embedder(model_name: str) -> Embedder:
    key = f"emb:{model_name}"
    if key not in _CACHE:
        _CACHE[key] = Embedder(model_name, batch_size=CONFIG.EMBED_BATCH_SIZE)
    return _CACHE[key]


def get_vector_store(index_dir_str: str) -> VectorStore | None:
    key = f"vs:{index_dir_str}"
    if key not in _CACHE:
        p = Path(index_dir_str)
        _CACHE[key] = VectorStore.load(p) if index_exists(p) else None
    return _CACHE[key]


def get_reranker(model_name: str) -> Reranker:
    key = f"rr:{model_name}"
    if key not in _CACHE:
        _CACHE[key] = Reranker(model_name)
    return _CACHE[key]


class PipelineError(RuntimeError):
    pass


def current_api_key() -> str | None:
    return CONFIG.OPENROUTER_API_KEY or st.session_state.get("pasted_api_key")


def llm_ready() -> bool:
    return bool(current_api_key())


def run_retrieval(question, store, embedder, top_k, use_reranker, top_n_rerank):
    if len(store) == 0:
        raise PipelineError("Index is empty. Run: python build_index.py")
    if use_reranker:
        pool = max(top_n_rerank, top_k)
        candidates = retrieve(question, store, embedder, pool)
        if not candidates:
            return [], False
        reranker = get_reranker(CONFIG.RERANK_MODEL)
        return reranker.rerank(question, candidates, top_k)[: CONFIG.MAX_CONTEXT_CHUNKS], True
    return retrieve(question, store, embedder, top_k)[: CONFIG.MAX_CONTEXT_CHUNKS], False


def answer_question(question, model, top_k, use_reranker, top_n_rerank) -> dict[str, Any]:
    store = get_vector_store(str(CONFIG.resolved_index_dir()))
    if store is None:
        raise PipelineError("Index not built. Run: python build_index.py")
    embedder = get_embedder(CONFIG.EMBEDDING_MODEL)

    started = time.perf_counter()
    contexts, reranked = run_retrieval(
        question, store, embedder, top_k, use_reranker, top_n_rerank
    )
    prompt = build_prompt(question, contexts)

    api_key = current_api_key()
    if api_key:
        try:
            answer = generate_answer(
                prompt=prompt, model=model, temperature=CONFIG.LLM_TEMPERATURE,
                provider=CONFIG.LLM_PROVIDER,
                api_key=api_key, base_url=CONFIG.OPENROUTER_BASE_URL,
            )
        except Exception as exc:
            answer = f"Generation failed: {exc}"
    else:
        answer = "Generation disabled -- no OpenRouter key. Retrieval chunks are below."

    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "answer": answer or "(empty answer)",
        "prompt": prompt,
        "contexts": contexts,
        "sources": cited_contexts(answer, contexts),
        "citations": parse_citations(answer, num_contexts=len(contexts)),
        "latency_ms": latency_ms,
        "retrieved": len(contexts),
        "reranked": reranked,
    }


def render_sources(sources):
    with st.expander("Sources"):
        if not sources:
            st.caption("No chunks retrieved.")
            return
        for position, chunk in enumerate(sources, start=1):
            title = (
                f"[{position}] {chunk.get('source','?')} "
                f"- page {chunk.get('page','?')} "
                f"- score {chunk.get('score',0.0):.3f}"
            )
            with st.expander(title):
                cap = f"chunk_id {chunk.get('chunk_id','-')} | cosine {chunk.get('score',0.0):.4f}"
                if "rerank_score" in chunk:
                    cap += f" | rerank {chunk['rerank_score']:.4f}"
                st.caption(cap)
                st.write(chunk.get("text", ""))


def render_turn(result, show_sources, show_prompt):
    st.write(result["answer"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Latency (ms)", str(max(1, int(round(result["latency_ms"])))))
    c2.metric("Retrieved chunks", result["retrieved"])
    c3.metric("Reranked", "yes" if result["reranked"] else "no")
    if show_sources:
        render_sources(result["sources"])
    if show_prompt:
        with st.expander("Prompt sent to the LLM"):
            st.code(result["prompt"], language=None)


def render_history(show_sources, show_prompt):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_turn(message, show_sources, show_prompt)


def render_sidebar(store):
    st.sidebar.title("Document Q&A")

    if not CONFIG.OPENROUTER_API_KEY:
        pasted = st.sidebar.text_input(
            "OpenRouter API key", type="password", key="api_key_input",
            help="Paste your sk-or-... key, or set OPENROUTER_API_KEY in .env",
        )
        st.session_state.pasted_api_key = (pasted or "").strip()

    default_model = CONFIG.LLM_MODEL if CONFIG.LLM_MODEL in LLM_MODELS else LLM_MODELS[0]
    model = st.sidebar.selectbox("Model", LLM_MODELS, index=LLM_MODELS.index(default_model))
    top_k = st.sidebar.slider("top_k", 1, 10, CONFIG.TOP_K)
    use_reranker = st.sidebar.toggle("Enable reranking", value=True)
    top_n_rerank = CONFIG.TOP_N_RERANK
    if use_reranker:
        top_n_rerank = st.sidebar.slider(
            "top_n_rerank", min_value=top_k, max_value=50,
            value=max(CONFIG.TOP_N_RERANK, top_k),
        )
    show_sources = st.sidebar.checkbox("Show retrieved chunks", value=True)
    show_prompt = st.sidebar.checkbox("Show LLM prompt", value=False)

    st.sidebar.divider()
    if st.sidebar.button("Reload index"):
        for k in [k for k in _CACHE if k.startswith(("vs:", "emb:"))]:
            del _CACHE[k]
        st.rerun()

    if store is not None:
        stats = store.stats()
        st.sidebar.caption(
            f"{stats['vectors']} chunks from {stats['sources']} document(s)\n\n"
            f"Embeddings: {CONFIG.EMBEDDING_MODEL}\n\n"
            f"Reranker: {CONFIG.RERANK_MODEL if use_reranker else 'disabled'}"
        )
    else:
        st.sidebar.caption(f"Embeddings: {CONFIG.EMBEDDING_MODEL}")

    return {
        "model": model, "top_k": top_k, "use_reranker": use_reranker,
        "top_n_rerank": top_n_rerank,
        "show_sources": show_sources, "show_prompt": show_prompt,
    }


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon=None, layout="centered")
    store = get_vector_store(str(CONFIG.resolved_index_dir()))
    settings = render_sidebar(store)

    st.header("Ask questions about your documents")

    if store is None:
        st.error("No index found.")
        st.info("Run `python build_index.py` first, then click 'Reload index'.")
        st.stop()

    if not llm_ready():
        st.warning("No OpenRouter credential. Paste a key in the sidebar or set OPENROUTER_API_KEY in .env.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_history(settings["show_sources"], settings["show_prompt"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption("Retrieving and generating...")
        try:
            result = answer_question(
                question=question, model=settings["model"],
                top_k=settings["top_k"],
                use_reranker=settings["use_reranker"],
                top_n_rerank=settings["top_n_rerank"],
            )
        except PipelineError as exc:
            result = {
                "answer": str(exc), "prompt": "", "contexts": [], "sources": [],
                "citations": [], "latency_ms": 0.0, "retrieved": 0, "reranked": False,
            }
        placeholder.empty()
        render_turn(result, settings["show_sources"], settings["show_prompt"])

    st.session_state.messages.append({"role": "assistant", **result})


if __name__ == "__main__":
    main()