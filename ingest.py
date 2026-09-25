"""
ingest.py

One-time build script. Run this from the project root whenever you:
  - Add or update PDFs in data/
  - Change your chunk size / overlap settings
  - Switch embedding models

Usage:
    python ingest.py

What it does:
    1. Loads all PDFs from data/
    2. Filters out near-empty pages (title/divider slides)
    3. Splits pages into chunks
    4. Embeds chunks and upserts to Pinecone

Do NOT import this file from app.py — it is a build step, not runtime code.
"""

from src.pdf_loader import load_all_pdfs
from src.chunker import split_documents
from src.vector_store import save_to_vector_store

# Pages with fewer characters than this are almost certainly
# title slides or section dividers — skip them so they don't
# pollute the vector store with near-empty chunks.
MIN_PAGE_LENGTH = 100


def filter_empty_pages(documents: list) -> list:
    """
    Removes near-empty pages (title slides, divider slides, image-only slides).
    These produce very short chunks that waste vectors and confuse retrieval.
    """
    before = len(documents)
    filtered = [doc for doc in documents if len(doc.page_content.strip()) >= MIN_PAGE_LENGTH]
    removed = before - len(filtered)
    if removed:
        print(f"  Filtered out {removed} near-empty page(s) (< {MIN_PAGE_LENGTH} chars)")
    return filtered


def main():
    print("\n=== RAG Ingestion Pipeline ===\n")

    # ── Step 1: Load PDFs ──────────────────────────────────────────────────────
    print("[1/3] Loading PDFs...")
    documents = load_all_pdfs()

    # ── Step 2: Filter + Chunk ─────────────────────────────────────────────────
    print("\n[2/3] Filtering and chunking...")
    documents = filter_empty_pages(documents)
    chunks = split_documents(documents)

    # Quick sanity check — print a sample chunk so you can verify quality
    print("\n--- Sample chunk (first) ---")
    sample = chunks[0]
    print(f"  Source : {sample.metadata.get('source')}")
    print(f"  Page   : {sample.metadata.get('page')}")
    print(f"  Length : {len(sample.page_content)} chars")
    print(f"  Preview: {sample.page_content[:200].strip()}...")
    print("----------------------------\n")

    # ── Step 3: Embed + Upsert to Pinecone ────────────────────────────────────
    print("[3/3] Embedding chunks and upserting to Pinecone...")
    save_to_vector_store(chunks)

    print("\n=== Ingestion complete ✓ ===")
    print("You can now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
