"""Graph nodes for the first vertical slice."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from agent.chat_model import make_chat_model
from agent.graph.state import AgentState
from agent.logging_config import get_logger
from agent.retrieval.retriever import DEFAULT_K, K_WIDEN_STEP, ToolRetriever
from agent.resilience import apply_limiter, with_retry

_log = get_logger(__name__)

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


def _goal(state: AgentState) -> str:
    return f"{state.get('plan', '')} {state['task']}".strip()


def make_retrieve_node(retriever: ToolRetriever):
    """Create a retrieval node closed over the current registry."""

    def retrieve_node(state: AgentState) -> dict:
        k = state.get("retrieval_k", DEFAULT_K)
        names = retriever.retrieve(_goal(state), k)
        _log.info(
            "retrieved tool subset",
            extra={"tools": names},
        )
        return {"available_tool_names": names}

    return retrieve_node


def widen_node(state: AgentState) -> dict:
    """Widen retrieval after an out-of-subset tool request."""
    remove_messages = []
    if state["messages"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and last.id:
            remove_messages.append(RemoveMessage(id=last.id))

    return {
        "retrieval_k": state.get("retrieval_k", DEFAULT_K) + K_WIDEN_STEP,
        "retrieval_miss_count": state.get("retrieval_miss_count", 0) + 1,
        "messages": remove_messages,
    }


def make_act_node(tools: list[Any]):
    """Create an act node that binds only the retrieved tool subset."""
    llm = make_chat_model()
    tool_by_name = {tool.name: tool for tool in tools}

    async def act_node(state: AgentState) -> dict:
        available = state.get("available_tool_names", [])
        subset = [tool_by_name[name] for name in available if name in tool_by_name]
        if not subset:
            subset = tools
        await apply_limiter()
        bound = llm.bind_tools(subset)
        response = await with_retry(lambda: bound.ainvoke(state["messages"]))
        return {"messages": [response]}

    return act_node
