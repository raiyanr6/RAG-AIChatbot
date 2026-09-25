"""
src/chatbot.py

Central orchestrator for the RAG pipeline.
Pinecone is persistent and hosted, so there's no local/in-memory
distinction anymore — the index is simply queried, not built at startup.
"""

from src.embedder import embed_text
from src.retriever import retrieve_chunks
from src.llm import generate_answer
from src.rate_limiter import check_rate_limit
from src.cache import check_cache, store_in_cache


def load_chatbot():
    """
    Kept for interface compatibility with app.py — no setup work needed
    anymore since Pinecone connects on demand, not at startup.
    """
    print("\n  Chatbot ready (Pinecone-backed, no startup build required).\n")
    return {}


def ask(chatbot: dict, question: str, session_id: str) -> dict:
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "pages": []}

    # 1. Rate limit — cheapest check, do it first, before spending anything
    if not check_rate_limit(session_id):
        return {"answer": "Rate limit exceeded. Please try again later.", "sources": [], "pages": []}

    # 2. Embed once — reused for both cache check and retrieval
    embedding = embed_text(question)

    # 3. Semantic cache check
    cached_answer = check_cache(embedding)
    if cached_answer:
        return {"answer": cached_answer, "sources": [], "pages": []}

    # 4. Cache miss — real retrieval + generation
    chunks = retrieve_chunks(embedding)
    answer = generate_answer(question, chunks)

    # 5. Cache the new answer for next time
    store_in_cache(question, answer, embedding)

    # 6. Collect source citations from the chunks already retrieved —
    #    no second retrieval call needed, unlike the original code
    source_pages = {}
    for chunk in chunks:
        source = chunk["source"]
        page = chunk["page"]
        source_pages.setdefault(source, [])
        if page not in source_pages[source]:
            source_pages[source].append(page)

    sources = list(source_pages.keys())
    pages = [sorted(source_pages[s]) for s in sources]

    return {"answer": answer, "sources": sources, "pages": pages}