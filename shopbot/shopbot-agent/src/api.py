# src/api.py
# Source: Book 2, Chapters 2–9
# Book 2 API endpoint — integrates the full Book 2 pipeline:
#   cache wrapper (Ch. 7) → query decomposition (Ch. 5) → hybrid retrieval (Ch. 3)
#   → cross-encoder rerank (Ch. 4) → memory-aware scoring (Ch. 6)
#   → multi-intent prompt builder (Ch. 5) → LLM → session update (Ch. 6)
#
# Run this file to serve the Book 2 system: python src/api.py
# The /health and /ask routes mirror Book 1 for drop-in compatibility.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import os
import re
import sys
import uvicorn

# Ensure the shopbot root is on sys.path when running as `python src/api.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.wrapper import answer_with_cache
from memory.session import load_session, save_session, Turn
from memory.update import push_turn, maybe_refresh_summary, extract_mentioned_skus

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cross-encoder reranking (Ch. 4) always runs in the rerank microservice —
    # a separate standalone project, ../shopbot-rerank/, run via local Docker
    # in dev, same as production. RERANK_URL is required — retrieval/rerank_client.py
    # raises KeyError at import time if it's unset, so a missing env var fails
    # before this even runs. Here we additionally confirm the service is
    # actually reachable, so "forgot to `docker run` the rerank container"
    # fails at startup with a clear error instead of on the first customer's
    # /ask request.
    import httpx
    from retrieval.rerank_client import RERANK_URL
    try:
        httpx.get(f"{RERANK_URL}/health", timeout=5.0).raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Rerank service unreachable at {RERANK_URL} — start it first "
            f"(../shopbot-rerank/: docker build -t shopbot-rerank . && "
            f"docker run -p 8001:8001 shopbot-rerank). Cause: {e}"
        ) from e

    _mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if _mlflow_uri:
        try:
            import mlflow
            import mlflow.openai
            mlflow.set_tracking_uri(_mlflow_uri)
            mlflow.set_experiment("shopbot-eval")
            mlflow.openai.autolog()
            print(f"  MLflow tracing → {_mlflow_uri}  (experiment: shopbot-eval)")
        except Exception as e:
            print(f"  MLflow tracing skipped: {e}")

    print("ShopBot v2 ready (hybrid + rerank + memory + cache).")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopBot API v2",
    description="Book 2 production pipeline — hybrid search, reranking, memory, cache",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response Models ─────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question:   str
    session_id: str = "default"

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        v = v.replace("‘", "'").replace("’", "'")   # smart single quotes
        v = v.replace("“", '"').replace("”", '"')   # smart double quotes
        v = v.replace("—", " - ")                        # em dash
        v = v.replace("​", "")                           # zero-width space

        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError(f"Question too long ({len(v)} chars). Maximum is 500.")

        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"you\s+are\s+now\s+",
            r"act\s+as\s+",
            r"system\s+prompt",
            r"forget\s+(everything|all)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid question format.")
        return v


class AnswerResponse(BaseModel):
    answer:     str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ShopBot", "version": "2.0.0"}


@app.post("/ask", response_model=AnswerResponse)
def ask_shopbot(payload: QuestionRequest):
    session = load_session(payload.session_id)

    answer, retrieved, _ = answer_with_cache(payload.question, session)

    # Update session memory with this turn.
    # Extract SKUs via style codes in the answer (not all retrieved chunks)
    # so that background sub-query chunks don't pollute active_skus.
    mentioned_skus = extract_mentioned_skus(retrieved, answer=answer)
    push_turn(session, Turn(role="user",      content=payload.question), [])
    push_turn(session, Turn(role="assistant", content=answer),           mentioned_skus)
    # Build a lightweight full-history approximation for the summary trigger
    maybe_refresh_summary(session, session.recent_turns)
    save_session(session)

    return AnswerResponse(answer=answer, session_id=payload.session_id)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    print("ShopBot v2 API starting...")
    print(f"  API:     http://localhost:{port}")
    print(f"  Swagger: http://localhost:{port}/docs")
    print(f"  Health:  http://localhost:{port}/health")
    uvicorn.run("src.api:app", host=host, port=port, reload=True)
