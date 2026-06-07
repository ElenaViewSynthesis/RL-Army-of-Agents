"""Build a code inspection subagent."""
from __future__ import annotations

from typing import Any, List
import logging


def build_code_agent(model: str, tools: List[Any]):
    system_message = (
        "You are a codebase inspection subagent. Analyse project structure, architecture, "
        "dependencies, and missing tests. Prefer concrete file paths and implementation-level recommendations.\n"
        "Do not modify files unless explicitly instructed."
    )

    try:
        from langchain.agents import create_agent
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("langchain not available, returning stub code agent: %s", exc)

        class _Stub:
            def __init__(self, system_message: str):
                self.system_message = system_message

            def invoke(self, payload: dict):
                return {"messages": [{"role": "assistant", "content": "stub codebase analysis"}]}

            async def ainvoke(self, payload: dict):
                return self.invoke(payload)

        return _Stub(system_message)

    agent = create_agent(model, tools)

    class AgentWrapper:
        def __init__(self, agent: Any, system_message: str):
            self._agent = agent
            self._system_message = system_message

        def invoke(self, payload: dict):
            messages = payload.get("messages", [])
            messages = [{"role": "system", "content": self._system_message}] + messages
            payload = {"messages": messages}
            if hasattr(self._agent, "invoke"):
                return self._agent.invoke(payload)
            return {"messages": [{"role": "assistant", "content": "agent missing invoke"}]}

        async def ainvoke(self, payload: dict):
            messages = payload.get("messages", [])
            messages = [{"role": "system", "content": self._system_message}] + messages
            payload = {"messages": messages}
            if hasattr(self._agent, "ainvoke"):
                return await self._agent.ainvoke(payload)
            if hasattr(self._agent, "invoke"):
                return self._agent.invoke(payload)
            return {"messages": [{"role": "assistant", "content": "agent missing ainvoke/invoke"}]}

    return AgentWrapper(agent, system_message)
