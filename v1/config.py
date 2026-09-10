"""
config.py
---------
Single place for API keys, model names, and file locations.
Nothing else in the app should hardcode these — change them here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------
# Read explicitly (rather than letting each library guess env var names)
# so a missing key fails immediately with a clear message here, instead
# of as a confusing error deep inside langchain/google-genai later.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

_missing = [
    name for name, val in [("GROQ_API_KEY", GROQ_API_KEY), ("GOOGLE_API_KEY", GOOGLE_API_KEY)]
    if not val
]
if _missing:
    raise RuntimeError(
        f"Missing required API key(s): {', '.join(_missing)}. "
        f"Create a .env file next to config.py (copy .env.example) and set them there. "
        f"GOOGLE_API_KEY is your Gemini key from https://aistudio.google.com/apikey "
        f"(not a GCP service-account file)."
    )

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "gemini-embedding-001"

# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------
# DATA_DIR is where the knowledge base lives (this is the ONLY thing
# that needs to survive a restart in this app):
#   - data/books/         mining textbook PDFs you want baked into the bot
#   - data/base_index/    the FAISS index built from those books
#
# Chat history is intentionally NOT stored here — see backend.py.
#
# Locally this defaults to a "data" folder next to this file.
# On Render, set the env var DATA_DIR=/app/data and mount a Persistent
# Disk there so your books/index survive restarts/redeploys (see README.md).
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))

BOOKS_DIR = DATA_DIR / "books"
BASE_INDEX_DIR = DATA_DIR / "base_index"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
