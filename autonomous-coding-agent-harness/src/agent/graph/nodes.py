"""Graph nodes for the first vertical slice."""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.graph.state import AgentState

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a precise coding agent. Use tools when needed, answer only "
        "from observed tool results, and stop when the task is complete."
    )
)


def plan_node(state: AgentState) -> dict:
    """Record the task as the initial plan."""
    return {"plan": state["task"]}


def retrieve_node(state: AgentState) -> dict:
    """Slice 1 retrieval: expose all discovered tools."""
    return {"available_tool_names": [tool.name for tool in state["tools"]]}


async def act_node(state: AgentState) -> dict:
    """Bind tools to the model and ask it for the next action."""
    model = os.environ.get("AGENT_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(model=model).bind_tools(state["tools"])
    messages = [SYSTEM_PROMPT, HumanMessage(content=state["task"])]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}
