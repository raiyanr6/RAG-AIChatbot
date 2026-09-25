import redis
import numpy as np
import hashlib
import os
from dotenv import load_dotenv
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
from openai import OpenAI
from redis.commands.search.index_definition import IndexDefinition, IndexType

load_dotenv()

CACHE_SIMILARITY_THRESHOLD = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", 0.92))

r = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
    password=os.environ["REDIS_PASSWORD"],
    decode_responses=False,  # note: False here, unlike rate_limiter.py — embeddings are binary bytes, not text
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def create_cache_index():
    try:
        r.ft("cache_idx").info()
        print("Index already exists, skipping creation.")
    except Exception:
        schema = (
            TextField("question"),
            TextField("answer"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
            }),
        )
        r.ft("cache_idx").create_index(
            schema,
            definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
        )
        print("Index created.")


def store_in_cache(question: str, answer: str, embedding: list[float]):
    vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
    key = f"cache:{hashlib.md5(question.encode()).hexdigest()}"
    r.hset(key, mapping={"question": question, "answer": answer, "embedding": vec_bytes})


def check_cache(embedding: list[float], threshold: float = CACHE_SIMILARITY_THRESHOLD) -> str | None:
    vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
    q = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("answer", "score")
        .dialect(2)
    )
    results = r.ft("cache_idx").search(q, query_params={"vec": vec_bytes})
    if results.docs:
        similarity = 1 - float(results.docs[0].score)
        print(f"  (closest match similarity: {similarity:.4f})")
        if similarity >= threshold:
            return results.docs[0].answer.decode() if isinstance(results.docs[0].answer, bytes) else results.docs[0].answer
    return None

