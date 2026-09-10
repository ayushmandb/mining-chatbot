"""
backend.py
----------
The LangGraph side of things: chat state, the single chat node, and tool
routing.

Chat memory is intentionally IN-MEMORY ONLY (LangGraph's MemorySaver):
each browser session gets its own thread_id (see app.py), and the
conversation for that thread lives only in this process's RAM. Close the
site (or the server restarts/redeploys) and it's gone — nothing about a
conversation is ever written to disk. That's on purpose: you don't get a
"past conversations" list, but you also never accumulate chat logs you
have to think about cleaning up.

The mining knowledge base (data/base_index/, built from data/books/) is
completely separate and unaffected by any of this — see rag_store.py.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import GROQ_API_KEY, GROQ_MODEL
from tools import tools

# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

MINING_SYSTEM_PROMPT = """You are "MineMind", an AI assistant that ONLY answers
questions related to the mining industry: mining engineering, geology,
mineral processing, mine safety, mine planning and design, equipment,
exploration, mining regulations, sustainability/ESG in mining, and mining
economics.

Tool rules:
1. For any question that could be answered from documents (the built-in
   mining books, or a PDF the user uploaded in this chat), call `rag_tool`
   FIRST.
2. If `rag_tool` doesn't return a useful answer, or the question is about
   something current (recent news, prices, regulation changes, live
   events), call `web_search`.
3. You may use both tools, one at a time, if a question needs it.
4. Never invent facts a tool could confirm. Base your final answer on what
   the tools actually returned, and say so if neither tool had an answer.

Scope rule:
- If the question has no reasonable connection to mining, politely decline,
  say you're a mining-focused assistant, and ask if they have a mining
  question instead. Do not answer unrelated questions "just this once".
"""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    messages = [SystemMessage(content=MINING_SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------
# Graph — MemorySaver keeps each thread_id's messages in RAM for as long
# as this process is running. Nothing here touches disk.
# ---------------------------------------------------------------------
checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)
