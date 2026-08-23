from langchain_text_splitters import RecursiveCharacterTextSplitter
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def split_documents(documents: list) -> list:
    """
    Takes a list of LangChain Document objects (one per PDF page) and
    splits them into smaller chunks.

    Returns a list of Document chunks, each inheriting the parent's metadata
    (source filename and page number).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # tries paragraph → line → word
    )

    chunks = splitter.split_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")
    return chunks