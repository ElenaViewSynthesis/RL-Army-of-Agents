"""Minimal graph spine for the first working agent slice."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import act_node, plan_node, retrieve_node
from agent.graph.state import AgentState


def build_graph(tools: list):
    """Build a minimal plan -> retrieve -> act -> tools graph."""
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
        lambda state: "tools" if state["messages"][-1].tool_calls else "end",
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", END)
    return graph.compile()
