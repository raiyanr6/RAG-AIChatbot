"""
app.py
Streamlit UI for the PDF chatbot.
Run with:  streamlit run app.py
"""

import streamlit as st
from src.chatbot import load_chatbot, ask

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lecture Notes Chatbot",
    page_icon="📚",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📚 Lecture Notes Chatbot")
st.caption("WIF3009 · Python for Scientific Computing")
st.markdown("Ask a question about your lecture notes or pick one below.")
st.divider()

# ── Initialise chatbot once ────────────────────────────────────────────────────
if "chain" not in st.session_state:
    with st.spinner("Loading chatbot and embedding notes... (first load ~30 seconds)"):
        st.session_state.chain = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# ── Render conversation history ────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources"):
                for source, page in zip(message["sources"], message["pages"]):
                    st.caption(f"• {source} — page {page}")

# ── Quick Questions ────────────────────────────────────────────────────────────
# ✅ Fixed: questions now match your actual ML lecture content
if not st.session_state.messages:
    st.markdown("### 💡 Quick Questions")
    sample_questions = [
        "What is the difference between classification and regression?",
        "How does the k-Nearest Neighbors algorithm work?",
        "What is overfitting and underfitting?",
        "What is K-Fold cross validation?",
    ]

    col1, col2 = st.columns(2)
    for i, q in enumerate(sample_questions):
        col = col1 if i % 2 == 0 else col2
        if col.button(q, use_container_width=True):
            st.session_state.selected_question = q
            st.rerun()

# ── Handle user input ──────────────────────────────────────────────────────────
chat_input_val = st.chat_input("Ask a question about your lectures...")
question = chat_input_val or st.session_state.selected_question

if question:
    st.session_state.selected_question = None

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching lecture notes..."):
            response = ask(st.session_state.chain, question)

        answer  = response["answer"]
        sources = response["sources"]
        pages   = response["pages"]

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources"):
                for source, page in zip(sources, pages):
                    st.caption(f"• {source} — page {page}")

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": sources,
        "pages"  : pages,
    })

    st.rerun()