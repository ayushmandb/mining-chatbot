# Mining Chatbot

A mining-domain-only chatbot (LangGraph + Streamlit) with:

- A shared knowledge base built from your own mining textbooks (PDFs)
- An optional per-chat PDF upload (used *in addition to* the shared books, only for that chat)
- Exactly two tools: search the knowledge base (`rag_tool`), or search the web (`web_search`)
- A system prompt that refuses non-mining questions
- Persistent, named chat threads (sqlite-backed)

## Project structure

```
mining-chatbot/
├── app.py                   # Streamlit UI — run this
├── backend.py                # LangGraph graph, system prompt, thread persistence
├── tools.py                  # the 2 tools: rag_tool, web_search
├── rag_store.py               # knowledge base: shared books + per-thread uploads
├── config.py                  # API keys, model names, file paths
├── build_knowledge_base.py   # one-off script: index data/books/ -> data/base_index/
├── data/
│   ├── books/                 # <- put your 3-4 mining PDFs here
│   ├── base_index/            # generated — the FAISS index (gitignored)
│   └── chatbot.db             # generated — chat history + thread titles (gitignored)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
└── render.yaml                # optional Render blueprint
```

## 1. Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your real keys
```

Your `.env` needs:
```
GROQ_API_KEY=...
GOOGLE_API_KEY=...      # this is your Gemini key — langchain-google-genai reads this exact var name
```

## 2. Build the shared mining knowledge base

Drop 3-4 mining textbook PDFs into `data/books/`, then:

```bash
python build_knowledge_base.py
```

This builds `data/base_index/` once. Every chat uses it automatically.
Re-run this script any time you add or remove books. (It calls the Gemini
embeddings API for every chunk, so it's not something you want running on
every app restart — that's why it's a separate script, not part of `app.py`.)

> If you skip this step, the app still runs — it just tells the user no
> knowledge base is loaded yet, and falls back to web search / general
> knowledge for mining questions.

## 3. Run it

```bash
streamlit run app.py
```

## How the two tools work

- **`rag_tool`** — searches the shared books, plus that specific chat's
  uploaded PDF if there is one. The tool automatically knows which chat
  it's being called from (via LangGraph's config, not by asking the LLM
  to remember a thread ID), so it can't mix up documents between chats.
- **`web_search`** — DuckDuckGo search, no API key needed, for anything
  current the books wouldn't have (prices, news, regulation changes).

The system prompt tells the model to always try `rag_tool` first for
document-answerable questions, fall back to `web_search` for current
info, and to decline anything unrelated to mining.

## Docker

```bash
docker build -t mining-chatbot .
docker run -p 8501:8501 --env-file .env mining-chatbot
```

Open http://localhost:8501

If you want your books baked into the image (simplest for a small
personal project), just make sure `data/books/*.pdf` isn't excluded by
`.gitignore`/`.dockerignore` before building — they aren't, by default.
Only `data/base_index/` and the sqlite db are ignored, since those are
regenerated automatically.

**Important:** if you commit copyrighted textbooks to a *public* GitHub
repo, that's on you to check the license/rights for — GitHub isn't private
storage. If your books aren't yours to redistribute, keep the repo private,
or don't commit `data/books/`, and instead copy the PDFs onto the server
some other way (e.g. via a Render persistent disk, see below).

## Deploying to Render

1. Push this repo to GitHub.
2. In Render: **New → Web Service**, connect the repo, environment = **Docker**.
3. Add environment variables: `GROQ_API_KEY`, `GOOGLE_API_KEY`.
4. Deploy.

### About "keeping memory on the server" (the thing you asked about)

Render's web services have an **ephemeral filesystem by default** — anything
written to disk (the sqlite chat-history db, the FAISS index) survives while
the instance is running, but is **wiped on every restart or redeploy**
(scale-to-zero, deploys, crashes, etc.).

Two options:

**Option A — Render Persistent Disk (recommended, needs a paid instance type):**
Add a disk mounted at `/app/data` (already wired up in `render.yaml`, or add
it manually in the Render dashboard under your service → **Disks**). Set
`DATA_DIR=/app/data` (already the default in the Dockerfile). Now
`data/books/`, `data/base_index/`, and `data/chatbot.db` all persist across
restarts and redeploys — chat history and the knowledge base survive
permanently, exactly like running it on your own machine.

**Option B — No persistent disk (Free plan):**
Everything in `DATA_DIR` resets on restart. This is fine for demoing, but:
- Chat history disappears after a restart.
- If you baked your books into the Docker image, the base index just
  rebuilds itself automatically the first time someone asks a question
  after a restart (via the fallback in `rag_store.get_base_store()`) —
  it'll just be a little slow on that first request, and re-calls the
  Gemini embeddings API each time, which has its own rate limits/cost.

If you want real persistence without paying for a disk, an alternative is
pointing `CHAT_DB_PATH`/`BASE_INDEX_DIR` at an external service instead of
local disk (e.g. a small hosted Postgres or S3-compatible bucket) — that's
a bigger change than "fix what you can quickly," so it's left out here, but
`config.py` is the only place you'd need to touch to redirect storage paths.

## Notes on what changed from your original version

- Removed the unused `metadata` variable and debug `print()` calls in the
  chat node.
- `rag_tool` no longer relies on the LLM correctly passing a `thread_id`
  argument — it's injected automatically from LangGraph's config, so it
  can't be forgotten or hallucinated by the model.
- Split one big backend file into `config.py` / `rag_store.py` / `tools.py`
  / `backend.py`, each with one job, while keeping `app.py` as the single
  thing you actually run.
- Added a shared, pre-built knowledge base (`data/books/` →
  `build_knowledge_base.py` → `data/base_index/`) alongside the
  per-chat upload feature you already had.
- Thread list now shows real titles (from the first message) and is
  stored in its own sqlite table instead of just listing raw UUIDs.
- System prompt now hard-scopes the bot to mining topics and reduced the
  tool count from an implicit "web + rag" to a strict 2-tool policy with
  explicit ordering (docs first, web second).
