"""Factory helpers to create and load MCP clients and tools."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, List

from app.config.settings import Settings
from app.mcp.server_configs import build_mcp_server_configs


async def create_mcp_client(settings: Settings) -> Any:
    """Create and start a MultiServerMCPClient or return a graceful stub.

    This attempts to start the client if it exposes an async `start`/`startup`
    method. Returns a started client or a lightweight stub when adapters are
    not available.
    """
    configs = build_mcp_server_configs(settings)
    logger = logging.getLogger(__name__)
    logger.info("Creating MCP client for servers: %s", ", ".join(configs.keys()))

    try:
        # import inside function so module import remains lightweight for tests
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(configs)

        # Try to start the client if it provides a startup method
        for start_name in ("start", "startup", "ainit", "init"):
            start = getattr(client, start_name, None)
            if start is not None:
                try:
                    result = start()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception("MCP client start method %s failed", start_name)
                break

        return client
    except Exception as exc:  # pragma: no cover - fallback path used when library missing
        logger.warning("langchain-mcp-adapters not available: %s", exc)

        class _StubClient:
            def __init__(self, configs):
                self.configs = configs

            async def get_tools(self):
                return []

            # no-op lifecycle methods to make shutdown idempotent
            async def stop(self):
                return None

            async def close(self):
                return None

            async def shutdown(self):
                return None

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


async def close_mcp_client(client: Any) -> None:
    """Attempt to gracefully stop/shutdown the MCP client if supported."""
    logger = logging.getLogger(__name__)
    if client is None:
        return

    for stop_name in ("shutdown", "stop", "close", "terminate"):
        stop = getattr(client, stop_name, None)
        if stop is not None:
            try:
                result = stop()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Failed to call client.%s()", stop_name)
            break
