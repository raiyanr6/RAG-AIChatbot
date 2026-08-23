import os
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_all_pdfs() -> list:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"data/ directory not found at {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    all_documents = []

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Normalise metadata so downstream code has consistent keys
        for page in pages:
            page.metadata["source"] = pdf_path.name  # e.g. "Lecture1.pdf"
            # PyPDFLoader already sets page.metadata["page"] (0-indexed int)
        all_documents.extend(pages)

    print(f"  Total pages loaded: {len(all_documents)}")
    return all_documents