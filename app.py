"""
app.py
Streamlit UI for the FSKTM Core Course Study Chatbot.
Run with:  streamlit run app.py
"""

import uuid
import streamlit as st
from src.chatbot import load_chatbot, ask

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FSKTM Study Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6B7280;
        font-size: 0.95rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    div[data-testid="stChatMessage"] {
        padding: 0.5rem 0;
    }
    .quick-q-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        text-align: left;
        padding: 0.6rem 1rem;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: #4F46E5;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 FSKTM Study Chatbot")
    st.caption("Ask questions across your core course lecture notes — grounded in the actual material, with sources cited.")
    st.divider()
    st.markdown("**Currently covering:**")
    st.markdown("- WIF3009 · Python for Scientific Computing")
    st.caption("More core courses coming soon.")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Built by Raiyan Rahman · [GitHub](https://github.com/raiyanr6)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 FSKTM Study Chatbot</h1>
    <p>Ask a question about your lecture notes — answers are grounded in course material, with sources cited.</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Session state initialisation ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chatbot" not in st.session_state:
    with st.spinner("Connecting to knowledge base..."):
        st.session_state.chatbot = load_chatbot()

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
                for source, page_list in zip(message["sources"], message["pages"]):
                    pages_str = ", ".join(str(p) for p in page_list)
                    st.caption(f"• {source} — pages {pages_str}")

# ── Quick Questions ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<p class="quick-q-label">💡 Try asking</p>', unsafe_allow_html=True)
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
    st.divider()

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
            response = ask(
                st.session_state.chatbot,
                question,
                st.session_state.session_id,
            )

        answer = response["answer"]
        sources = response["sources"]
        pages = response["pages"]

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources"):
                for source, page_list in zip(sources, pages):
                    pages_str = ", ".join(str(p) for p in page_list)
                    st.caption(f"• {source} — pages {pages_str}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "pages": pages,
    })

    st.rerun()