"""
src/vector_store.py

Two modes:
  - save_to_vector_store()        → persists to disk   (local dev via ingest.py)
  - build_inmemory_vector_store() → stays in RAM       (Streamlit Cloud deployment)
  - load_vector_store()           → loads from disk    (local dev via app.py)
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"


def save_to_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embeds chunks and persists to disk.
    Called by ingest.py for local development.
    """
    if not chunks:
        raise ValueError("No chunks provided.")

    print(f"  Saving {len(chunks)} chunks to Chroma at: {CHROMA_DIR}")

    if CHROMA_DIR.exists():
        print("  Existing chroma_db found — clearing before rebuild...")
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="lecture_notes",
    )

    print(f"  Done. Collection holds {vector_store._collection.count()} vectors.")
    return vector_store


def build_inmemory_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embeds chunks and stores in RAM only.
    Uses EphemeralClient explicitly for Streamlit Cloud compatibility.
    """
    if not chunks:
        raise ValueError("No chunks provided.")

    import chromadb
    client = chromadb.EphemeralClient()  # ✅ explicit in-memory client

    print(f"  Building in-memory Chroma with {len(chunks)} chunks...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="lecture_notes",
        client=client,               # ✅ pass client explicitly
    )

    print(f"  Done. Collection holds {vector_store._collection.count()} vectors.")
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Loads a persisted Chroma store from disk.
    Used locally — NOT used in Streamlit Cloud deployment.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"chroma_db/ not found at {CHROMA_DIR}.\n"
            "Run `python ingest.py` first to build the vector store."
        )

    print(f"  Loading Chroma store from: {CHROMA_DIR}")

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="lecture_notes",
    )

    count = vector_store._collection.count()
    if count == 0:
        raise ValueError("Chroma store is empty. Re-run `python ingest.py`.")

    print(f"  Loaded. Collection holds {count} vectors.")
    return vector_store