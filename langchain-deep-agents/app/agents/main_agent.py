"""Build the main orchestration agent that coordinates subagents and MCP tools."""
from __future__ import annotations

from typing import Any, List
import logging


def build_main_agent(model: str, tools: List[Any]):
    system_message = (
        "You are the main software architecture agent.\n\n"
        "Your job is to coordinate MCP tools and specialised subagents to produce a clear engineering report.\n\n"
        "You can:\n"
        "- use GitHub tools to inspect repository metadata\n"
        "- use filesystem tools to inspect local files\n"
        "- use PostgreSQL tools to inspect database metadata\n"
        "- delegate research to the research subagent\n"
        "- delegate codebase analysis to the codebase subagent\n"
        "- delegate database analysis to the database subagent\n\n"
        "Follow this workflow:\n"
        "1. Understand the user request.\n"
        "2. Decide which tools or subagents are needed.\n"
        "3. Gather evidence from available tools.\n"
        "4. Avoid destructive operations.\n"
        "5. Produce a structured report.\n\n"
        "Your final answer must include:\n"
        "- Executive summary\n"
        "- Repository architecture\n"
        "- MCP tools used\n"
        "- Database findings\n"
        "- Missing tests\n"
        "- Risks\n"
        "- Recommended next implementation steps\n"
    )

    try:
        from langchain.agents import create_agent
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("langchain not available, returning stub main agent: %s", exc)

        class _Stub:
            def __init__(self, system_message: str):
                self.system_message = system_message

            def invoke(self, payload: dict):
                return {"messages": [{"role": "assistant", "content": "stub main orchestration result"}]}

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
