"""
app.py
------
Streamlit frontend for the mining chatbot.

Chat memory lasts only for the current browser session: closing the site
(or the server restarting) clears it. There's no "past conversations"
list, on purpose — see backend.py for why.

Run locally with:
    streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import chatbot
from rag_store import (
    has_base_knowledge_base,
    ingest_user_pdf,
    thread_document_metadata,
    thread_has_own_document,
)


def start_new_chat():
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message_history"] = []


st.set_page_config(page_title="Mining Chatbot", page_icon="⛏️")

if "thread_id" not in st.session_state:
    start_new_chat()

thread_id = st.session_state["thread_id"]

# ============================ Sidebar ============================
st.sidebar.title("⛏️ Mining Chatbot")

if st.sidebar.button("➕ New chat", use_container_width=True):
    start_new_chat()
    st.rerun()

st.sidebar.caption("Chat memory lasts for this visit only — it clears when you close the site.")

st.sidebar.divider()
st.sidebar.subheader("Knowledge base")

if has_base_knowledge_base():
    st.sidebar.success("Built-in mining books loaded ✅")
else:
    st.sidebar.warning(
        "No built-in mining books indexed yet.\n\n"
        "Add PDFs to `data/books/` and run `python build_knowledge_base.py`."
    )

if thread_has_own_document(thread_id):
    meta = thread_document_metadata(thread_id)
    st.sidebar.info(
        f"Plus your upload for this chat: `{meta.get('filename')}` "
        f"({meta.get('chunks')} chunks, {meta.get('documents')} pages)"
    )

uploaded_pdf = st.sidebar.file_uploader(
    "Add your own mining PDF to THIS chat (optional)", type=["pdf"]
)
if uploaded_pdf and not thread_has_own_document(thread_id):
    with st.sidebar.status("Indexing your PDF…", expanded=True) as box:
        ingest_user_pdf(uploaded_pdf.getvalue(), thread_id, uploaded_pdf.name)
        box.update(label="✅ PDF indexed for this chat", state="complete", expanded=False)
    st.rerun()

# ============================ Main chat ===========================
st.title("Mining Assistant")
st.caption("Ask about mining engineering, geology, safety, equipment, and more.")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a mining question…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": thread_id}}

    with st.chat_message("assistant"):
        status_box = {"box": None}

        def stream_answer():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    name = getattr(chunk, "name", "tool")
                    if status_box["box"] is None:
                        status_box["box"] = st.status(f"🔧 Using `{name}`…", expanded=False)
                    else:
                        status_box["box"].update(label=f"🔧 Using `{name}`…")
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        try:
            answer = st.write_stream(stream_answer())
        except Exception as e:
            answer = "⚠️ Something went wrong while answering."
            st.error(str(e))

        if status_box["box"] is not None:
            status_box["box"].update(label="✅ Done", state="complete")

    st.session_state["message_history"].append({"role": "assistant", "content": answer})
