# main.py
# Source: Book 2, Chapter 4 + Chapter 9 — Real Production
# Standalone rerank microservice. Runs as its own container so the cross-encoder
# CPU work is isolated from ../shopbot-agent's API process and scales independently
# — no shared code, dependencies, or deploy lifecycle with that project.
#
# shopbot-agent calls this via HTTP only (retrieval/rerank_client.py) and has no
# in-process fallback — this service is a required dependency, not optional.
# Two workers (--workers 2) saturate a 2-core box; scale horizontally for more.

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
app = FastAPI(title="ShopBot Rerank Service", version="2.0.0")


class RerankRequest(BaseModel):
    query:      str
    candidates: list[dict]    # [{"id": str, "text": str}]
    top_k:      int = 3


class RerankResponse(BaseModel):
    ranked_ids: list[str]
    scores:     list[float]


@app.get("/health")
def health():
    return {"status": "ok", "service": "rerank"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.candidates:
        return RerankResponse(ranked_ids=[], scores=[])
    pairs  = [(req.query, c["text"]) for c in req.candidates]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(req.candidates, scores), key=lambda p: -p[1])
    top    = ranked[:req.top_k]
    return RerankResponse(
        ranked_ids=[c["id"] for c, _ in top],
        scores=[round(s, 4) for _, s in top],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=2)
