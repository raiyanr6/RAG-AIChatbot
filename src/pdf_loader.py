import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# This works both locally and on Streamlit Cloud
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "data"


def load_all_pdfs() -> list:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"data/ directory not found at {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    # ── TEMPORARY DEBUG — remove after confirming ──────────────────
    print(f"  DATA_DIR resolved to: {DATA_DIR}")
    print(f"  PDFs found: {[f.name for f in pdf_files]}")
    # ───────────────────────────────────────────────────────────────

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    print(f"  Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_documents = []

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        all_documents.extend(pages)

    print(f"  Total pages loaded: {len(all_documents)}")
    return all_documents