"""Factory helpers to create and load MCP clients and tools."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, List

from app.config.settings import Settings
from app.mcp.server_configs import build_mcp_server_configs


async def create_mcp_client(settings: Settings) -> Any:
    """Create a MultiServerMCPClient or return a graceful stub when unavailable.

    The function is async to allow future async initialisation of real clients.
    """
    configs = build_mcp_server_configs(settings)
    logger = logging.getLogger(__name__)
    logger.info("Creating MCP client for servers: %s", ", ".join(configs.keys()))

    try:
        # import inside function so module import remains lightweight for tests
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(configs)
        return client
    except Exception as exc:  # pragma: no cover - fallback path used when library missing
        logger.warning("langchain-mcp-adapters not available: %s", exc)

        class _StubClient:
            def __init__(self, configs):
                self.configs = configs

            async def get_tools(self):
                return []

        return _StubClient(configs)


async def load_mcp_tools(settings: Settings) -> List[Any]:
    """Load tools from all configured MCP servers.

    Returns an empty list if tools cannot be loaded.
    """
    logger = logging.getLogger(__name__)
    client = await create_mcp_client(settings)

    get_tools = getattr(client, "get_tools", None)
    if get_tools is None:
        logger.warning("MCP client has no get_tools method; returning empty tool list")
        return []

    try:
        result = get_tools()
        if inspect.isawaitable(result):
            tools = await result
        else:
            tools = result
    except Exception as exc:
        logger.exception("Failed to fetch MCP tools: %s", exc)
        tools = []

    logger.info("Loaded %d MCP tools", len(tools) if tools else 0)
    return tools
