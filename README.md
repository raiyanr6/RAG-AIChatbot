# 📚 AI Lecture Notes Chatbot — RAG Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that allows students to query university lecture notes using natural language. Built with LangChain, ChromaDB, OpenAI, and Streamlit.

🔗** https://rag-aichatbot-raiyan.streamlit.app/**

---

## 📌 What It Does

Instead of manually searching through lecture slides, students can ask questions in plain English and get accurate answers — grounded strictly in the lecture content, with source citations showing exactly which PDF and page the answer came from.

```
Student: "What is the difference between classification and regression?"
    ↓
Chatbot searches 700+ embedded lecture chunks
    ↓
Retrieves the 4 most relevant passages
    ↓
GPT-4o-mini generates a grounded answer
    ↓
Answer displayed with source citations (PDF + page number)
```

---

## 🏗️ Architecture

```
PDF Lecture Notes
       ↓
PDF Loader (PyPDF)
       ↓
Text Chunker (RecursiveCharacterTextSplitter)
       ↓
Embedding Model (Sentence Transformers — all-MiniLM-L6-v2)
       ↓
Vector Store (ChromaDB — in-memory)
       ↓
User Question → Similarity Search → Top-K Chunks
       ↓
Prompt Template + GPT-4o-mini
       ↓
Answer + Source Citations (Streamlit UI)
```

---

## 🛠️ Tech Stack

| Component        | Tool                                      |
|-----------------|-------------------------------------------|
| Language         | Python 3.11                              |
| LLM              | OpenAI GPT-4o-mini                       |
| Framework        | LangChain                                |
| Embeddings       | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database  | ChromaDB (in-memory for deployment)      |
| PDF Parsing      | PyPDF                                    |
| UI               | Streamlit                                |
| Deployment       | Streamlit Cloud                          |

---

## 📁 Project Structure

```
RAGPipeline/
├── src/
│   ├── pdf_loader.py      # Loads and normalises PDF documents
│   ├── chunker.py         # Splits documents into overlapping chunks
│   ├── embedder.py        # HuggingFace embedding model
│   ├── vector_store.py    # ChromaDB (disk + in-memory modes)
│   ├── retriever.py       # Similarity search over vector store
│   ├── llm.py             # OpenAI LLM + RAG prompt template
│   └── chatbot.py         # Orchestrates full pipeline
├── data/                  # Lecture PDF files
├── ingest.py              # One-time ingestion script (local dev)
├── app.py                 # Streamlit UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone the repository
```bash
git clone https://github.com/your-username/RAGPipeline.git
cd RAGPipeline
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Add your PDFs
Place your lecture PDF files inside the `data/` folder.

### 6. Run the app
```bash
streamlit run app.py
```

The app will load and embed all PDFs in memory on first startup (~30 seconds), then open at `http://localhost:8501`.

---

## 💡 Key Design Decisions

**Hallucination prevention** — The prompt template explicitly instructs the LLM to answer only from the retrieved context. If the answer isn't in the lecture notes, the chatbot says so rather than making something up.

**In-memory vector store** — ChromaDB runs fully in memory for Streamlit Cloud deployment, requiring zero external infrastructure. PDFs are re-embedded on each cold start (~30 seconds).

**Modular architecture** — Each concern (loading, chunking, embedding, retrieval, LLM, UI) is isolated in its own module. Swapping any component (e.g. replacing OpenAI with a local model, or ChromaDB with Pinecone) requires changing only one file.

**Source citations** — Every answer includes the source PDF filename and page number so students can verify answers and read further.

---

## ⚠️ Known Limitations

- **Code snippets not indexed** — Lecture slides with code embedded as images (screenshots) are not extracted by PyPDF and therefore cannot be retrieved
- **Cold start time** — First load takes ~30 seconds on Streamlit Cloud while PDFs are embedded in memory
- **Question sensitivity** — Retrieval quality depends on how closely the question wording matches the lecture text. Rephrasing a question can improve results

---

## 🔮 Future Improvements

- [ ] Persistent vector store (Pinecone) to eliminate cold start delay
- [ ] OCR support for code-screenshot slides
- [ ] Conversation memory for follow-up questions
- [ ] Query rewriting to improve retrieval on ambiguous questions
- [ ] Support for uploading custom PDFs via the UI

---
