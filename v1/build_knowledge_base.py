"""
build_knowledge_base.py
------------------------
Run this once, and again whenever you add/remove books:

    1. Put your mining textbook PDFs inside data/books/
    2. Run: python build_knowledge_base.py
    3. It creates data/base_index/ — the FAISS index the app loads on
       startup and shares across every chat.

This is intentionally a separate script from the web app itself, so
indexing a few big textbooks doesn't slow down (or cost embedding-API
calls on) every single app restart.
"""

from config import BOOKS_DIR
from rag_store import build_base_index

if __name__ == "__main__":
    pdfs = list(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {BOOKS_DIR}. Add your mining book PDFs there first.")
    else:
        print(f"Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")
        print("Building index (this calls the Gemini embeddings API, "
              "so it can take a while for big books)...")
        build_base_index(force=True)
        print(f"Done. Index saved to {BOOKS_DIR.parent / 'base_index'}")
