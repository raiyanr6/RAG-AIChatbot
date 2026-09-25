"""
src/vector_store.py

Two functions:
  - save_to_vector_store() → embeds chunks and upserts to Pinecone (called by ingest.py)
  - get_vector_store()     → connects to the existing Pinecone index (called by app.py/chatbot.py)

No local disk, no in-memory mode — Pinecone is persistent and hosted
"""

import os
from pinecone import Pinecone, ServerlessSpec
from src.embedder import embed_documents, EMBEDDING_DIM

PINECONE_INDEX_NAME = "rag-lecture-notes"

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def _ensure_index_exists():
    existing = [idx["name"] for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"  Index '{PINECONE_INDEX_NAME}' not found — creating...")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"  Index '{PINECONE_INDEX_NAME}' already exists.")


def save_to_vector_store(chunks: list) -> None:
    """
    Embeds chunks and upserts them to Pinecone.
    Called by ingest.py — the only place writes happen.
    """
    if not chunks:
        raise ValueError("No chunks provided.")

    _ensure_index_exists()
    index = _pc.Index(PINECONE_INDEX_NAME)

    print(f"  Embedding {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_documents(texts)

    print(f"  Upserting {len(chunks)} vectors to Pinecone...")
    vectors = [
        (
            f"chunk_{i}",
            embedding,
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", -1),
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Batch upserts — Pinecone recommends batches of ~100 to avoid oversized requests
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    stats = index.describe_index_stats()
    print(f"  Done. Index holds {stats['total_vector_count']} vectors.")


def get_vector_store():
    """
    Connects to the existing Pinecone index — no embedding, no ingestion.
    Called by chatbot.py/retriever.py at query time.
    """
    _ensure_index_exists()
    return _pc.Index(PINECONE_INDEX_NAME)