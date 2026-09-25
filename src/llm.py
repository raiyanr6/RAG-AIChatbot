"""
src/llm.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 512

RAG_PROMPT_TEMPLATE = """
You are a helpful teaching assistant for a university course called
"Python for Scientific Computing".

Use ONLY the context below to answer the student's question.
If the answer is not contained in the context, say:
"I don't have enough information in the lecture notes to answer that."
Do not make up information or use knowledge outside the provided context.

Context:
{context}

Student's question: {question}

Answer:
""".strip()


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        import streamlit as st
        api_key = api_key or st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found.\n"
            "Add it to your .env file:  OPENAI_API_KEY=sk-..."
        )
    return api_key


_client = OpenAI(api_key=_get_api_key())


def format_chunks(chunks: list[dict]) -> str:
    """Joins retrieved chunk texts into a single context string."""
    return "\n\n".join(chunk["text"] for chunk in chunks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generates an answer from the question + retrieved chunks.
    Called directly by chatbot.py — no chain object, no retriever
    passed in, since retrieval already happened before this is called.
    """
    context = format_chunks(chunks)

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"  Answer generated ({LLM_MODEL})")
    return response.choices[0].message.content