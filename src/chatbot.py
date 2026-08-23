"""
src/chatbot.py

Central orchestrator for the RAG pipeline.
Supports two modes:
  - Local dev   : loads from disk (fast startup)
  - Deployment  : builds in-memory on startup (no disk needed)
"""

from src.pdf_loader import load_all_pdfs
from src.chunker import split_documents
from src.embedder import get_embedding_model
from src.vector_store import build_inmemory_vector_store, load_vector_store
from src.llm import build_rag_chain

# Set to True for Streamlit Cloud deployment
# Set to False for local development (requires ingest.py to have been run)
USE_INMEMORY = True

MIN_PAGE_LENGTH = 100


def _filter_empty_pages(documents: list) -> list:
    return [doc for doc in documents if len(doc.page_content.strip()) >= MIN_PAGE_LENGTH]


def load_chatbot():
    """
    Initialises the full RAG pipeline and returns
    a dict containing the chain and retriever.
    """
    print("\n  Initialising chatbot...")

    embeddings = get_embedding_model()

    if USE_INMEMORY:
        # ── Deployment mode: build everything in-memory ──────────────────────
        print("  Mode: in-memory (deployment)")
        documents = load_all_pdfs()
        documents = _filter_empty_pages(documents)
        chunks    = split_documents(documents)
        vector_store = build_inmemory_vector_store(chunks, embeddings)
    else:
        # ── Local mode: load from persisted chroma_db/ ───────────────────────
        print("  Mode: disk (local)")
        vector_store = load_vector_store(embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = build_rag_chain(retriever)

    print("  Chatbot ready.\n")
    return {"chain": chain, "retriever": retriever}


def ask(chatbot: dict, question: str) -> dict:
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "pages": []}

    chain     = chatbot["chain"]
    retriever = chatbot["retriever"]

    answer   = chain.invoke(question)
    src_docs = retriever.invoke(question)

    # Collect all pages per source file
    source_pages = {}
    for doc in src_docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", 0) + 1

        if source not in source_pages:
            source_pages[source] = []
        if page not in source_pages[source]:
            source_pages[source].append(page)

    # Build display lists
    sources = list(source_pages.keys())
    pages   = [sorted(source_pages[s]) for s in sources]

    return {"answer": answer, "sources": sources, "pages": pages}