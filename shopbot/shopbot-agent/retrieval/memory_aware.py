# retrieval/memory_aware.py
# Source: Book 2, Chapter 6 — ShopBot Finally Remembers
# Memory-aware retrieval: applies an active-SKU bonus on top of cross-encoder scores
# so the reranker slightly favours products already in the conversation context.
#
# Also applies the Chapter 8 cross-encoder confidence floor: if no candidate
# scores above 0.55, the retriever returns [] and the API routes to support.
# This replaces Book 1's 0.75 cosine-similarity threshold, which was calibrated
# against a 5-product catalog; the floor is calibrated against the labeled dataset.
#
# Style-code bypass: the cross-encoder (MS MARCO) scores literal SKU queries low
# because it doesn't understand style codes as identifiers. Any chunk whose text
# contains the exact style code from the query is given a floor override so it
# passes through even if the CE score is below 0.55.

import re
from infra.types import Chunk
from retrieval.hybrid import retrieve_hybrid
from retrieval.rerank_client import rerank_scores_remote

ACTIVE_SKU_BOOST        = 0.05   # tie-breaker; not enough to override a clearly better chunk
RERANK_CONFIDENCE_FLOOR = 0.55   # Chapter 8: calibrated against labeled_dataset.jsonl

_STYLE_CODE_RE = re.compile(r'\b[A-Z]+-\d+[A-Z]?\b')


def _style_codes(text: str) -> set[str]:
    return set(_STYLE_CODE_RE.findall(text))


def retrieve_for_prompt_with_memory(query: str, session) -> list[Chunk]:
    """
    Hybrid retrieval (top-20) → cross-encoder rerank with active-SKU bonus → top 3.
    Chunks below the confidence floor are withheld (same gate-kept spirit as Book 1).
    """
    candidates = retrieve_hybrid(query, top_n=20)
    if not candidates:
        return []

    ce_scores = rerank_scores_remote(query, candidates)

    active_skus = getattr(session, "active_skus", []) if session is not None else []
    query_codes = _style_codes(query)

    boosted: list[tuple[Chunk, float]] = []
    for c, s in zip(candidates, ce_scores):
        sku   = c.metadata.get("product_id") or c.metadata.get("sku", "")
        bonus = ACTIVE_SKU_BOOST if sku in active_skus else 0.0
        score = float(s) + bonus
        # Style-code bypass: MS MARCO cross-encoder scores literal SKU queries low.
        # If the chunk contains the exact style code from the query, lift to floor.
        if query_codes and query_codes & _style_codes(c.text):
            score = max(score, RERANK_CONFIDENCE_FLOOR)
        boosted.append((c, score))

    boosted.sort(key=lambda item: item[1], reverse=True)

    # Apply Chapter 8 confidence floor
    above_floor = [(c, s) for c, s in boosted if s >= RERANK_CONFIDENCE_FLOOR]

    # When the session has active products, return only chunks from those products
    # if any pass the floor. This prevents sibling products from entering the
    # prompt on follow-up queries (e.g. other coord-sets when asking about size L
    # for a specific co-ord set the customer is already discussing).
    # Only fall back to unrestricted top-3 if no active-sku chunks cleared the floor.
    if active_skus:
        active_chunks = [
            c for c, _ in above_floor
            if (c.metadata.get("product_id") or c.metadata.get("sku", "")) in active_skus
        ]
        if active_chunks:
            return active_chunks[:3]
        # Active products had no chunks above the floor — fall through to normal top-3
    return [c for c, _ in above_floor[:3]]
