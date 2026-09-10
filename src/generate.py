from __future__ import annotations

import os
import re

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about documents. "
    "Use ONLY the provided context. Cite the context blocks you use with "
    "bracketed numbers, e.g. [1] or [2]. If the answer is not in the "
    "context, say so explicitly instead of guessing."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, contexts: list[dict]) -> str:
    if not contexts:
        block = "(no context was retrieved)"
    else:
        parts = []
        for i, ctx in enumerate(contexts, start=1):
            parts.append(
                f"[{i}] {ctx.get('source', '?')} "
                f"(page {ctx.get('page', '?')}):\n{ctx.get('text', '')}"
            )
        block = "\n\n".join(parts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{block}\n\n"
        f"Question: {question}\n\n"
        f"Answer with citations like [1], [2]:"
    )


def _make_client(api_key: str | None, base_url: str | None) -> OpenAI:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("No OpenRouter API key configured")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG Document Q&A",
        },
    )


def generate_answer(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    provider: str = "openrouter",
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    client = _make_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def parse_citations(answer: str, num_contexts: int | None = None) -> list[int]:
    if not answer:
        return []
    seen: list[int] = []
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if num_contexts is not None and not (1 <= n <= num_contexts):
            continue
        if n not in seen:
            seen.append(n)
    return seen


def cited_contexts(answer: str, contexts: list[dict]) -> list[dict]:
    nums = parse_citations(answer, num_contexts=len(contexts))
    picked = [contexts[n - 1] for n in nums if 1 <= n <= len(contexts)]
    return picked if picked else contexts