from functools import lru_cache
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.config import EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME, OLLAMA_HOST


def get_all_ingested_docs() -> dict:
    """Get all ingested documents with their metadata."""
    store = get_store()
    all_data = store.get()
    docs = {}
    for metadata in all_data["metadatas"]:
        doc_id = metadata.get("doc_id")
        if doc_id and doc_id not in docs:
            source = metadata.get("source", "unknown")
            docs[doc_id] = {"doc_id": doc_id, "source": source, "chunks": 0}

    for doc_id in docs:
        chunks = store.get(where={"doc_id": doc_id})
        docs[doc_id]["chunks"] = len(chunks["ids"])

    return docs


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def add_documents(docs: list[Document], doc_id: str) -> None:
    store = get_store()
    existing = store.get(where={"doc_id": doc_id})
    if existing["ids"]:
        store.delete(ids=existing["ids"])
    store.add_documents(docs)


def retriever(k: int, doc_id: str | None = None):
    store = get_store()
    search_kwargs = {"k": k}
    if doc_id:
        search_kwargs["filter"] = {"doc_id": doc_id}
    return store.as_retriever(search_kwargs=search_kwargs)
