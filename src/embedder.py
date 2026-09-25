"""
src/embedder.py
Creates an OpenAI embedding model client.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_embedding_model():
    """
    Returns a callable-style wrapper so calling code stays consistent
    regardless of provider (mirrors the old HuggingFaceEmbeddings interface
    just enough for embed_query/embed_documents-style calls).
    """
    return _client


def embed_text(text: str) -> list[float]:
    """Embed a single string. Used at query time (chatbot.py)."""
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of strings. Used at ingest time (ingest.py) — OpenAI's API
    accepts a list directly, so we batch instead of looping one-by-one,
    which is faster and reduces API call count.
    """
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]