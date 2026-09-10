"""
tools.py
--------
Exactly two tools, on purpose:

1. rag_tool   -> the mining knowledge base (built-in books + whatever PDF
                 the user uploaded in this specific chat, if any)
2. web_search -> the open web, for anything current the books won't have

Note on `config: RunnableConfig` in rag_tool: LangChain auto-injects this
argument at call time (it's excluded from the schema shown to the model),
so the LLM never has to know or guess the thread_id — LangGraph's ToolNode
passes the real one in automatically. That's a lot more reliable than
asking the model to pass thread_id itself.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag_store import get_retriever_for_thread

_web_search_engine = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Search the mining knowledge base for information relevant to the query.
    This includes the built-in mining reference books and any PDF the user
    has uploaded in this chat. Always try this tool first for mining
    questions that could be answered from documents.
    """
    thread_id = (config.get("configurable") or {}).get("thread_id")
    retriever = get_retriever_for_thread(thread_id)

    if retriever is None:
        return (
            "No knowledge base is available right now (no mining books have "
            "been indexed yet and the user hasn't uploaded a PDF). Mention "
            "this and answer from general mining knowledge if you reasonably can."
        )

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant passages were found in the knowledge base for this query."

    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current or recent information — news, prices,
    regulations, standards, incidents, equipment releases, etc.
    Only use this for mining-related queries.
    """
    return _web_search_engine.run(query)


tools = [rag_tool, web_search]
