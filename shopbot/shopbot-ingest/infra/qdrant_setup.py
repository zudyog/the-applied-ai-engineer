# infra/qdrant_setup.py
# Source: Book 2, Chapter 2 — Move Without Breaking Things
# Creates the Qdrant collection with a 1536-dim dense vector and a sparse BM25 field.
# Run once to bootstrap. Safe to re-run — checks before creating.
#
# Collection: "zudyog_catalog"
# Dense:  1536-dim cosine (same embedding model as Book 1 ChromaDB collection)
# Sparse: BM25 (populated by ingestion/bm25_backfill.py in Chapter 3)

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "zudyog_catalog"


def create_collection() -> None:
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # Declared empty in Chapter 2; BM25 values filled in Chapter 3.
            "bm25": models.SparseVectorParams(),
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")
    print("  dense: 1536-dim cosine — ready for ingestion/dual_write.py")
    print("  bm25:  sparse field declared — populate with ingestion/bm25_backfill.py")


if __name__ == "__main__":
    create_collection()
