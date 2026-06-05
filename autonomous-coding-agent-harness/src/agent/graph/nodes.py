"""Graph nodes for the first vertical slice."""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.graph.state import AgentState

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a precise coding agent. Execute the user's task exactly as "
        "stated, no more and no less. Use tools when needed. Your final answer "
        "must be grounded only in tool results visible in this session. If a "
        "tool fails, state the failure instead of inventing success. When the "
        "task is complete, stop immediately."
    )
)


def plan_node(state: AgentState) -> dict:
    """Record the task and seed the conversation."""
    return {
        "plan": state["task"],
        "messages": [SYSTEM_PROMPT, HumanMessage(content=state["task"])],
    }


def retrieve_node(state: AgentState) -> dict:
    """Slice 1 retrieval: expose all discovered tools."""
    return {"available_tool_names": [tool.name for tool in state["tools"]]}


async def act_node(state: AgentState) -> dict:
    """Bind tools to the model and ask it for the next action."""
    model = os.environ.get("AGENT_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(model=model).bind_tools(state["tools"])
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
