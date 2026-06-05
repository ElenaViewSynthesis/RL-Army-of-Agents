"""Minimal graph spine for the first working agent slice."""

from langchain_core.messages import AIMessage

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import act_node, plan_node, retrieve_node
from agent.graph.state import AgentState


def _should_continue(state: AgentState) -> str:
    """Continue to tools only when the model requested a tool call."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def build_graph(tools: list):
    """Build a minimal plan -> retrieve -> act -> tools -> act graph."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("act", act_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "act")
    graph.add_conditional_edges(
        "act",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "act")
    return graph.compile()
