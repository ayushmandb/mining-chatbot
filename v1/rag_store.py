"""
rag_store.py
------------
Everything about the mining knowledge base lives here. This file doesn't
know about the LLM or Streamlit at all — it just stores documents and
gives them back.

Two layers:

1. BASE store — one shared FAISS index built once from every PDF in
   data/books/ (your 3-4 mining textbooks). Every chat can use this.

2. PER-THREAD store — if a user uploads their own PDF in a specific
   chat, we build a copy of the base index + their document, scoped to
   that one thread only. Other chats never see it.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_INDEX_DIR, BOOKS_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
)

# Gemini's free tier allows 100 embedding requests/minute. Embedding a
# whole book in one call blows straight through that. Instead we embed in
# small batches with a pause between each, staying comfortably under the
# limit, and auto-retry (with a longer wait) if a batch still gets
# rate-limited.
_EMBED_BATCH_SIZE = 20
_EMBED_PAUSE_SECONDS = 15
_RATE_LIMIT_RETRY_WAIT = 60

# thread_id -> FAISS store that contains base books + that thread's own PDF
_THREAD_STORES: Dict[str, FAISS] = {}
_THREAD_METADATA: Dict[str, dict] = {}

_base_store: Optional[FAISS] = None
_base_load_attempted = False


def _load_pdf_chunks(path: Path):
    docs = PyPDFLoader(str(path)).load()
    return _TEXT_SPLITTER.split_documents(docs), len(docs)


def _embed_in_batches(chunks: List[Document], store: Optional[FAISS] = None) -> FAISS:
    """
    Embeds a list of chunks into a FAISS store a small batch at a time,
    pausing between batches to stay under the free-tier rate limit.
    If a batch still gets rate-limited (429), waits and retries that
    same batch instead of losing progress on the whole book.
    """
    total = len(chunks)
    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = chunks[start : start + _EMBED_BATCH_SIZE]
        while True:
            try:
                if store is None:
                    store = FAISS.from_documents(batch, embeddings)
                else:
                    store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print(
                        f"Hit the embedding rate limit at chunk {start}/{total}. "
                        f"Waiting {_RATE_LIMIT_RETRY_WAIT}s before retrying..."
                    )
                    time.sleep(_RATE_LIMIT_RETRY_WAIT)
                    continue
                raise
        done = min(start + _EMBED_BATCH_SIZE, total)
        print(f"Embedded {done}/{total} chunks...")
        if done < total:
            time.sleep(_EMBED_PAUSE_SECONDS)
    return store


def build_base_index(force: bool = False) -> Optional[FAISS]:
    """
    Build (or rebuild) the shared FAISS index from every PDF in data/books/.

    This calls the Gemini embeddings API for every chunk of every book, so
    it's meant to be run occasionally (via build_knowledge_base.py), not on
    every request. It's also called automatically as a fallback if no index
    exists yet on disk but books are present (e.g. first boot on a fresh
    Render instance).
    """
    global _base_store

    if BASE_INDEX_DIR.exists() and not force:
        _base_store = FAISS.load_local(
            str(BASE_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        return _base_store

    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        _base_store = None
        return None

    all_chunks = []
    for pdf in pdfs:
        chunks, _ = _load_pdf_chunks(pdf)
        all_chunks.extend(chunks)

    store = _embed_in_batches(all_chunks)
    BASE_INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(BASE_INDEX_DIR))
    _base_store = store
    return store


def get_base_store() -> Optional[FAISS]:
    """Lazily loads (or builds) the base store the first time it's needed."""
    global _base_store, _base_load_attempted
    if _base_store is None and not _base_load_attempted:
        _base_load_attempted = True
        build_base_index()
    return _base_store


def has_base_knowledge_base() -> bool:
    return get_base_store() is not None


def ingest_user_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """
    Index a user-uploaded PDF for ONE chat thread only. The resulting
    store contains the shared mining books (if any) PLUS this document,
    so the bot can answer from either.
    """
    if not file_bytes:
        raise ValueError("No file bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        chunks, num_pages = _load_pdf_chunks(tmp_path)
        user_store = _embed_in_batches(chunks)

        if BASE_INDEX_DIR.exists():
            # Fresh copy loaded from disk so we never mutate the shared
            # base store that other chat threads are using.
            combined = FAISS.load_local(
                str(BASE_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
            combined.merge_from(user_store)
        else:
            combined = user_store

        thread_id = str(thread_id)
        _THREAD_STORES[thread_id] = combined
        _THREAD_METADATA[thread_id] = {
            "filename": filename,
            "documents": num_pages,
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[thread_id]
    finally:
        tmp_path.unlink(missing_ok=True)


def get_retriever_for_thread(thread_id: Optional[str], k: int = 6):
    """
    Picks the right store for a thread:
      1. that thread's own store (base + their upload), if they made one
      2. otherwise the shared base store
      3. otherwise None (nothing indexed anywhere yet)
    """
    thread_id = str(thread_id) if thread_id else None
    store = _THREAD_STORES.get(thread_id) if thread_id else None
    if store is None:
        store = get_base_store()
    if store is None:
        return None
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def thread_has_own_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_STORES


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
