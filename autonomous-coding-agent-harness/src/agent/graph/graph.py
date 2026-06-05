"""Minimal graph spine for the first working agent slice."""

from langchain_core.messages import AIMessage

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import make_act_node, make_retrieve_node, plan_node, widen_node
from agent.graph.state import AgentState
from agent.retrieval.retriever import ToolRetriever


def _should_continue(state: AgentState) -> str:
    """Continue to tools only when the model requested a tool call."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        available = set(state.get("available_tool_names", []))
        if available:
            for call in last.tool_calls:
                if call["name"] not in available:
                    return "miss"
        return "tools"
    return "end"


def build_graph(tools: list, retriever: ToolRetriever):
    """Build a minimal plan -> retrieve -> act -> tools -> act graph."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("widen", widen_node)
    graph.add_node("act", make_act_node(tools))
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "act")
    graph.add_conditional_edges(
        "act",
        _should_continue,
        {"tools": "tools", "miss": "widen", "end": END},
    )
    graph.add_edge("widen", "retrieve")
    graph.add_edge("tools", "act")
    return graph.compile()
