"""
src/retriever.py
Loads the persisted Chroma vector store and retrieves relevant chunks.
Used for local development only — deployment uses chatbot.py in-memory mode.
"""

from langchain_chroma import Chroma
from src.embedder import get_embedding_model
from src.vector_store import load_vector_store

TOP_K = 4


def get_retriever():
    """
    Loads Chroma from disk and returns a retriever.
    Only used locally (requires ingest.py to have been run).
    """
    embeddings   = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    print(f"  Retriever ready (top_k={TOP_K})")
    return retriever


def retrieve_chunks(question: str, retriever) -> list:
    """
    Returns TOP_K most relevant chunks for a given question.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunks = retriever.invoke(question)

    print(f"\n  Retrieved {len(chunks)} chunks for: '{question}'")
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        print(f"    [{i+1}] {source} — page {page} ({len(chunk.page_content)} chars)")

    return chunks