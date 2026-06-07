"""Optional Deep Agents integration with fallback.

This module attempts to use `deepagents.create_deep_agent` when available, and
falls back to `create_agent` to preserve MVP functionality.
"""
from __future__ import annotations

from typing import Any, List
import logging


try:  # defensive import
    from deepagents import create_deep_agent  # type: ignore
except Exception:
    create_deep_agent = None  # pragma: no cover


def build_deep_software_agent(model: str, tools: List[Any]):
    """Create a deep-agent configured for planning and long-running analysis.

    If `deepagents` is missing, fall back to a regular agent.
    """
    system_message = (
        "Deep software agent for long-running repository analysis, planning, and delegated investigation."
    )

    if create_deep_agent is not None:
        try:
            return create_deep_agent(model=model, tools=tools, system_message=system_message)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("create_deep_agent failed: %s", exc)

    # Fallback to regular create_agent if deepagents is not present or fails.
    try:
        from langchain.agents import create_agent
        return create_agent(model, tools)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("Falling back to stub agent: %s", exc)

        class _Stub:
            def invoke(self, payload: dict):
                return {"messages": [{"role": "assistant", "content": "deep agent stub"}]}

            async def ainvoke(self, payload: dict):
                return self.invoke(payload)

        return _Stub()
