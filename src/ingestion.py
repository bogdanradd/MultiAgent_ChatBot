from pathlib import Path
import sys
import io
import pymupdf4llm
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.vectorstore import add_documents


def load_pdf(path: Path) -> list[Document]:
    docs = []

    # Suppress pymupdf4llm verbose output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        md_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    filename = path.name
    doc_id = path.stem

    for page_data in md_pages:
        page_num = page_data["metadata"]["page"]
        content = page_data["text"]
        docs.append(Document(
            page_content=content,
            metadata={"source": filename, "page": page_num, "doc_id": doc_id}
        ))

    return docs


def load_csv(path: Path) -> list[Document]:
    docs = []
    df = pd.read_csv(path)
    filename = path.name
    doc_id = path.stem

    for i, row in df.iterrows():
        content = " | ".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(
            page_content=content,
            metadata={"source": filename, "row": i, "doc_id": doc_id}
        ))

    return docs


def chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def ingest(path: str | Path) -> str:
    path = Path(path)
    doc_id = path.stem

    if path.suffix.lower() == ".pdf":
        docs = load_pdf(path)
        docs = chunk(docs)
    elif path.suffix.lower() == ".csv":
        docs = load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    add_documents(docs, doc_id)
    return doc_id
