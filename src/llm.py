"""
src/llm.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LLM_MODEL   = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS  = 512

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


def get_llm():
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
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
    )
    print(f"  LLM ready ({LLM_MODEL}, temperature={TEMPERATURE})")
    return llm


def format_docs(docs) -> str:
    """Joins retrieved chunk texts into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    llm = get_llm()

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Modern LCEL chain — replaces the old RetrievalQA
    chain = (
        {
            "context" : retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("  RAG chain ready")
    return chain