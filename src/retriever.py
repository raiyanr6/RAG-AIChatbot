"""
src/retriever.py
Retrieves relevant chunks from Pinecone given a precomputed question embedding.
Embedding happens in chatbot.py — this file only searches.
"""

from src.vector_store import get_vector_store

TOP_K = 4


def retrieve_chunks(embedding: list[float]) -> list[dict]:
    """
    Returns TOP_K most relevant chunks for a given question embedding.
    Expects an already-computed embedding, not raw text — chatbot.py
    is responsible for calling embed_text() first (shared with the
    semantic cache check).
    """
    index = get_vector_store()

    result = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    chunks = [
        {
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "unknown"),
            "page": match["metadata"].get("page", "?"),
            "score": match["score"],
        }
        for match in result["matches"]
    ]

    print(f"\n  Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"    [{i+1}] {chunk['source']} — page {chunk['page']} (score: {chunk['score']:.4f})")

    return chunks