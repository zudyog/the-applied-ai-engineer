# infra/clients.py
# Source: Book 2, Appendix A — Code Wiring
# Shared client module. Build once; import everywhere.
# Every module that needs embed(), llm, qdrant_client, bm25, reranker, or r
# imports from here — never constructs its own client.

import os
from openai import OpenAI
from qdrant_client import QdrantClient
import redis
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────────────────────────
_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed(text: str) -> list[float]:
    """OpenAI text-embedding-3-small → 1536-dim vector. Same model as Book 1."""
    resp = _openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


class _LLM:
    """Thin wrapper over chat.completions for the prose-style usage in chapters."""

    def __init__(self, model: str, temperature: float = 0):
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        *,
        system: str | None = None,
        user: str | None = None,
        prompt: str | None = None,
        max_tokens: int = 800,
    ) -> str:
        if prompt is not None and system is None and user is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if user:
                messages.append({"role": "user", "content": user})
        resp = _openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content


# Two LLM handles — the main answerer and the query decomposer (Chapter 5).
# Both use gpt-4o-mini in Book 2; Book 3 may move the decomposer to a cheaper model.
llm       = _LLM("gpt-4o-mini", temperature=0)
llm_small = _LLM("gpt-4o-mini", temperature=0)

# ── Qdrant Cloud ─────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# ── Redis — sessions + cache ─────────────────────────────────────────────────
r = redis.from_url(os.environ["REDIS_URL"])

# Cross-encoder reranking (Chapter 4) runs exclusively in the rerank microservice
# — a separate standalone project, ../shopbot-rerank/main.py — no in-process
# CrossEncoder here. See retrieval/rerank_client.py, which requires RERANK_URL
# and calls that service. This process never imports sentence-transformers/torch.

# ── BM25 sparse encoder — Chapter 3 ─────────────────────────────────────────
bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
