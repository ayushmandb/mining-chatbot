# ⛏️ Mining Chatbot

A mining-domain-only chatbot (LangGraph + Streamlit) with:

- A shared knowledge base built from your own mining textbooks (PDFs)
- An optional per-chat PDF upload (used *in addition to* the shared books, only for that one chat)
- Exactly two tools: search the knowledge base (`rag_tool`), or search the web (`web_search`)
- A system prompt that keeps the bot strictly on mining topics
- Multiple chats per browser visit — start new chats and switch back to older ones in the same session
- No login, no database — everything resets when the browser tab/site is closed

## Project structure

```
mining-chatbot/
├── app.py                    # Streamlit UI — run this
├── backend.py                 # LangGraph graph, system prompt, chat routing
├── tools.py                   # the 2 tools: rag_tool, web_search
├── rag_store.py                # knowledge base: shared books + per-thread uploads
├── config.py                   # API keys, model names, file paths
├── build_knowledge_base.py    # one-off script: index data/books/ -> data/base_index/
├── data/
│   ├── books/                  # <- put your mining PDFs here
│   └── base_index/             # generated — the FAISS index (gitignored)
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
GEMINI_API_KEY=...      # your Gemini key — langchain-google-genai reads GOOGLE_API_KEY internally via config.py
```

Get a Gemini key from https://aistudio.google.com/apikey (an API key, not a GCP service-account file).
Get a Groq key from https://console.groq.com/keys.

## 2. Build the shared mining knowledge base

Drop your mining textbook PDFs into `data/books/`, then:

```bash
python build_knowledge_base.py
```

This calls the Gemini embeddings API for every chunk of every book in `data/books/` and writes `data/base_index/`. **Every run rebuilds the index from scratch** — if you add a new book later, re-running this script re-embeds all books, old and new, not just the new one. Budget your embedding-API time/quota accordingly, especially on Gemini's free tier (rate-limited, so big books take a while — see the pacing constants in `rag_store.py` if this needs tuning).

> If you skip this step, the app still runs — it just tells the user no knowledge base is loaded yet, and falls back to web search / general knowledge for mining questions.

## 3. Run it

```bash
streamlit run app.py
```

## How chat memory works

- Each browser visit can hold **multiple chats** — "➕ New chat" starts a fresh one, and every chat you've started in this visit stays listed in the sidebar so you can switch back to it.
- All of this lives in Streamlit's `session_state` **and** in an in-memory LangGraph checkpointer (`MemorySaver`) on the server. Nothing about a conversation is ever written to disk.
- Closing the browser tab, or restarting/redeploying the server, wipes every chat permanently. There's no "past conversations across visits" — that would require a login system and a database, which this project intentionally doesn't have.

## How the two tools work

- **`rag_tool`** — searches the shared books, plus that specific chat's uploaded PDF if there is one. The tool automatically knows which chat it's being called from (via LangGraph's config, not by asking the LLM to remember an ID), so it can't mix up documents between chats.
- **`web_search`** — DuckDuckGo search, no API key needed, for anything current the books wouldn't have (prices, news, regulation changes).

The system prompt tells the model to always try `rag_tool` first for document-answerable questions, fall back to `web_search` for current info, and to decline anything unrelated to mining.

## Docker

```bash
docker build -t mining-chatbot .
docker run -p 8501:8501 --env-file .env mining-chatbot
```

Open http://localhost:8501

