"""Typed state carried through the agent graph."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    task: str
    plan: str
    tools: list[Any]
    available_tool_names: list[str]
    messages: list[BaseMessage]
