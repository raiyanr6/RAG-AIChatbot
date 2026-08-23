"""
src/embedder.py
Creates a HuggingFace sentence-transformer embedding model.
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible HuggingFace embedding model.
    Downloaded automatically on first use and cached locally.
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings