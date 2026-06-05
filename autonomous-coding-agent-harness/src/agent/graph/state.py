"""Typed state carried through the agent graph."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    task: str
    plan: str
    tools: list[Any]
    available_tool_names: list[str]
    retrieval_k: int
    retrieval_miss_count: int
    messages: Annotated[list[BaseMessage], add_messages]
