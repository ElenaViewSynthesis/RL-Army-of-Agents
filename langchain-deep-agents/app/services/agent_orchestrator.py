"""Orchestration service that initializes MCP tools, subagents and the main agent."""
from __future__ import annotations

import inspect
import logging
from typing import Any, List

from app.config.settings import Settings


class AgentOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.mcp_tools: List[Any] = []
        self.tavily_tools: List[Any] = []
        self.research_agent = None
        self.code_agent = None
        self.db_agent = None
        self.main_agent = None

    async def initialize(self) -> None:
        """Initialize MCP tools, subagents and the main orchestration agent."""
        # Create and start MCP client, then load tools
        try:
            from app.mcp.client_factory import create_mcp_client

            self.mcp_client = await create_mcp_client(self.settings)
            get_tools = getattr(self.mcp_client, "get_tools", None)
            if get_tools is None:
                self.logger.warning("MCP client has no get_tools method; continuing with empty tools")
                self.mcp_tools = []
            else:
                res = get_tools()
                if inspect.isawaitable(res):
                    self.mcp_tools = await res
                else:
                    self.mcp_tools = res
        except Exception as exc:
            self.logger.exception("Failed to load MCP tools: %s", exc)
            self.mcp_tools = []

        try:
            from app.tools.tavily_tools import build_tavily_tools

            self.tavily_tools = build_tavily_tools(self.settings)
        except Exception as exc:
            self.logger.exception("Failed to load Tavily tools: %s", exc)
            self.tavily_tools = []

        # Build subagents
        from app.agents.research_agent import build_research_agent
        from app.agents.code_agent import build_code_agent
        from app.agents.db_agent import build_db_agent

        research_tools = [*self.mcp_tools, *self.tavily_tools]
        self.research_agent = build_research_agent(self.settings.research_model, research_tools)
        self.code_agent = build_code_agent(self.settings.code_model, self.mcp_tools)
        self.db_agent = build_db_agent(self.settings.db_model, self.mcp_tools)

        # Wrap subagents as tools
        from app.tools.subagent_tools import (
            build_research_tool,
            build_codebase_tool,
            build_database_tool,
        )

        research_tool = build_research_tool(self.research_agent)
        code_tool = build_codebase_tool(self.code_agent)
        db_tool = build_database_tool(self.db_agent)

        # Compose full toolset: MCP tools first, then subagent tools
        tools: List[Any] = []
        if self.mcp_tools:
            tools.extend(self.mcp_tools)
        tools.extend([research_tool, code_tool, db_tool])

        # Build the main agent
        from app.agents.main_agent import build_main_agent

        self.main_agent = build_main_agent(self.settings.main_model, tools)

    async def shutdown(self) -> None:
        """Gracefully shutdown MCP client and any other resources."""
        try:
            from app.mcp.client_factory import close_mcp_client

            if getattr(self, "mcp_client", None) is not None:
                await close_mcp_client(self.mcp_client)
        except Exception as exc:
            self.logger.exception("Error while shutting down MCP client: %s", exc)

    async def run(self, user_query: str) -> str:
        """Run the main agent with the provided `user_query` and return its final message."""
        if not self.main_agent:
            raise RuntimeError("AgentOrchestrator not initialized. Call initialize() first.")

        payload = {"messages": [{"role": "user", "content": user_query}]}

        # Prefer async invocation
        if hasattr(self.main_agent, "ainvoke"):
            result = await self.main_agent.ainvoke(payload)
        elif hasattr(self.main_agent, "invoke"):
            result = self.main_agent.invoke(payload)
        else:
            raise RuntimeError("Main agent has no invoke/ainvoke method")

        # Extract the final assistant message
        try:
            from app.tools.subagent_tools import extract_final_message

            return extract_final_message(result)
        except Exception:
            return ""
