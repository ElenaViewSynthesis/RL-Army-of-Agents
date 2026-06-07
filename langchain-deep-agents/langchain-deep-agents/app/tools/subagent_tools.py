"""Wrap subagents as LangChain tools.

Each builder returns a callable that can be passed into `create_agent` as a tool.
The implementation is defensive so missing LangChain tooling libraries fall back to
lightweight decorators that keep the shape.
"""
from __future__ import annotations

from typing import Any, Callable
import asyncio
import logging


def extract_final_message(result: dict) -> str:
    try:
        messages = result.get("messages", [])
        if not messages:
            return ""
        return messages[-1].get("content", "")
    except Exception:
        return ""


def _safe_tool_decorator(name: str, description: str):
    try:
        from langchain.tools import tool as _real_tool

        return _real_tool(name=name, description=description)
    except Exception:
        # simple passthrough decorator for environments without langchain.tools
        def _decorator(fn: Callable):
            fn._tool_name = name
            fn._tool_description = description
            return fn

        return _decorator


def build_research_tool(research_agent: Any):
    _decorator = _safe_tool_decorator(
        "research_project_context", "Research a topic and return concise findings"
    )

    @_decorator
    def _tool(query: str) -> str:
        logger = logging.getLogger(__name__)
        try:
            # prefer async agent if available
            if hasattr(research_agent, "ainvoke"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # running in an event loop; try sync invoke if possible
                    if hasattr(research_agent, "invoke"):
                        result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
                    else:
                        return "Subagent requires async execution environment"
                else:
                    result = asyncio.run(research_agent.ainvoke({"messages": [{"role": "user", "content": query}]}))
            elif hasattr(research_agent, "invoke"):
                result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
            else:
                result = {"messages": [{"role": "assistant", "content": "no subagent available"}]}
        except Exception as exc:
            logger.exception("research tool invocation failed: %s", exc)
            result = {"messages": [{"role": "assistant", "content": f"error: {exc}"}]}

        return extract_final_message(result)

    return _tool


def build_codebase_tool(code_agent: Any):
    _decorator = _safe_tool_decorator(
        "inspect_codebase", "Inspect repository files and return analysis"
    )

    @_decorator
    def _tool(query: str) -> str:
        try:
            if hasattr(code_agent, "ainvoke"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    if hasattr(code_agent, "invoke"):
                        result = code_agent.invoke({"messages": [{"role": "user", "content": query}]})
                    else:
                        return "Subagent requires async execution environment"
                else:
                    result = asyncio.run(code_agent.ainvoke({"messages": [{"role": "user", "content": query}]}))
            elif hasattr(code_agent, "invoke"):
                result = code_agent.invoke({"messages": [{"role": "user", "content": query}]})
            else:
                result = {"messages": [{"role": "assistant", "content": "no subagent available"}]}
        except Exception as exc:
            logging.getLogger(__name__).exception("code tool invocation failed: %s", exc)
            result = {"messages": [{"role": "assistant", "content": f"error: {exc}"}]}

        return extract_final_message(result)

    return _tool


def build_web_graph_tool(web_graph_agent: Any):
    _decorator = _safe_tool_decorator(
        "ingest_web_knowledge_graph",
        "Search the web, extract pages, convert findings into graph entities, and store them in Neo4j",
    )

    @_decorator
    def _tool(query: str) -> str:
        try:
            if hasattr(web_graph_agent, "ainvoke"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    if hasattr(web_graph_agent, "invoke"):
                        result = web_graph_agent.invoke({"messages": [{"role": "user", "content": query}]})
                    else:
                        return "Subagent requires async execution environment"
                else:
                    result = asyncio.run(web_graph_agent.ainvoke({"messages": [{"role": "user", "content": query}]}))
            elif hasattr(web_graph_agent, "invoke"):
                result = web_graph_agent.invoke({"messages": [{"role": "user", "content": query}]})
            else:
                result = {"messages": [{"role": "assistant", "content": "no subagent available"}]}
        except Exception as exc:
            logging.getLogger(__name__).exception("web graph tool invocation failed: %s", exc)
            result = {"messages": [{"role": "assistant", "content": f"error: {exc}"}]}

        return extract_final_message(result)

    return _tool


def build_database_tool(db_agent: Any):
    _decorator = _safe_tool_decorator(
        "inspect_database", "Inspect database metadata and return findings"
    )

    @_decorator
    def _tool(query: str) -> str:
        try:
            if hasattr(db_agent, "ainvoke"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    if hasattr(db_agent, "invoke"):
                        result = db_agent.invoke({"messages": [{"role": "user", "content": query}]})
                    else:
                        return "Subagent requires async execution environment"
                else:
                    result = asyncio.run(db_agent.ainvoke({"messages": [{"role": "user", "content": query}]}))
            elif hasattr(db_agent, "invoke"):
                result = db_agent.invoke({"messages": [{"role": "user", "content": query}]})
            else:
                result = {"messages": [{"role": "assistant", "content": "no subagent available"}]}
        except Exception as exc:
            logging.getLogger(__name__).exception("database tool invocation failed: %s", exc)
            result = {"messages": [{"role": "assistant", "content": f"error: {exc}"}]}

        return extract_final_message(result)

    return _tool
