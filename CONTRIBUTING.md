# Contributing to The Applied AI Engineer

Thanks for considering a contribution. This repo is the real, working codebase behind Book 2 of the Zudyog RAG Mastery Series — it's meant to stay a runnable, honest reference implementation, not a polished framework. Contributions that keep it that way are the most welcome kind.

## Ground rules

- **Keep it runnable.** Every change should leave the four-project quickstart in [`README.md`](README.md) working from a clean clone: `shopbot-ingest` populates Qdrant, `shopbot-rerank` starts via Docker, `shopbot-agent` serves `/ask`, `zudyog-fashion` talks to it.
- **Keep it honest.** Two concrete rules this codebase already holds itself to — don't quietly loosen either in a PR:
  - The cross-encoder confidence floor (`RERANK_CONFIDENCE_FLOOR = 0.55` in `shopbot/shopbot-agent/retrieval/memory_aware.py`) stays where it is. A support fallback is correct behavior when nothing clears it — lowering the floor to reduce visible fallbacks reintroduces the type-mismatch failures Book 1 documented.
  - The RAGAS targets (Faithfulness 0.83, Answer Relevance 0.81, Context Precision 0.88, Context Recall 0.79) aren't claimed as "met" until the labeled dataset backing them is actually 100 queries, not the current 20. Don't pad the dataset with easy queries to move the number — sample from the failed-query log too.
  - If you think something structural should change instead, open an issue first rather than quietly "fixing" it in a PR.
- **Small PRs over big ones.** A focused PR that does one thing is easier to review and merge than a large one that touches many files — especially here, where a change to `shopbot-agent` might need a matching change in `shopbot-rerank`'s HTTP contract or `shopbot-ingest`'s Qdrant schema.

## Good first issues

If you're looking for a place to start, these are real gaps in the repo today — genuinely useful, and scoped small enough for a first PR:

1. **Add a GitHub Actions CI workflow.** There's currently no CI at all across any of the three Python projects — even a job that installs each `requirements.txt` and runs `python -m py_compile` (or a real smoke test, once #2 exists) on push/PR would catch broken dependency pins and import errors before they reach `main`.
2. **Add pytest unit tests for `shopbot-ingest/src/chunker.py`.** It's a pure function (`product_to_chunks()` — no API calls, no network), which makes it the easiest part of the pipeline to actually unit test — and right now it has zero test coverage anywhere in the repo.
3. **Add Dockerfiles for `shopbot-ingest` and `shopbot-agent`.** Only `shopbot-rerank` has one today. There's no containerized path to run the ingest pipeline or the API itself — for the API in particular, that's usually the first thing people reach for before setting up Railway.
4. **Expand `shopbot-agent/evaluation/labeled_dataset.jsonl`** from its current 20 queries toward the Chapter 8 spec of 100. Answer Relevance and Context Recall aren't statistically meaningful yet on 20 queries — and the dataset should include queries pulled from real failures, not just queries you already expect to pass.
5. **Add a `GET /products/{id}` endpoint** to `shopbot-agent/src/api.py`. The frontend's product pages (`zudyog-fashion/lib/products.ts`) currently read catalog data directly rather than through the API — a real product-detail endpoint would make the backend a complete, self-contained API surface, matching how `/ask` already works.

Open an issue to claim one before starting, so two people don't end up duplicating work.

## Reporting bugs

Please include:
- Which project it's in — `shopbot-ingest`, `shopbot-agent`, `shopbot-rerank`, or `zudyog-fashion` — each has its own `venv`/`.env`, so bugs are usually scoped to one
- What you ran (exact command) and what you expected vs. what happened
- Your Python/Node/Docker version
- Whether it reproduces on a clean clone (rules out local `venv`/`chroma_db`/cache state issues)

## Pull requests

1. Fork, branch off `main`
2. Make your change; keep commits focused
3. There's no automated test suite yet (see Good First Issues #1–2 above) — until there is, verify manually: start the affected project(s) per the Quickstart in `README.md` and confirm `/health` responds and a sample `/ask` query still returns a grounded answer
4. If you touched `shopbot-agent/retrieval/`, `shopbot-agent/cache/`, or the system prompt, run `python -m evaluation.evaluate` from `shopbot-agent/` and include the before/after scores in your PR description
5. Open the PR with a short description of *why*, not just *what* — the reasoning is what makes review fast

## Questions

Open a [discussion or issue](https://github.com/zudyog/the-applied-ai-engineer/issues) — no question is too basic.
