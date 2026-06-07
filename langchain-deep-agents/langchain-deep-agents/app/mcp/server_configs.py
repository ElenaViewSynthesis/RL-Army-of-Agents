"""Build MCP server configurations from settings."""
from __future__ import annotations

from typing import Any, Dict
import logging

from app.config.settings import Settings


def build_mcp_server_configs(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """Return a dict of MCP server configs.

    Validates presence of critical secrets and paths and raises ValueError with
    a clear message if something is missing.
    """
    missing = []
    if not settings.repo_path:
        missing.append("REPO_PATH")
    if not settings.github_personal_access_token:
        missing.append("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not settings.database_url:
        missing.append("DATABASE_URL")

    if missing:
        raise ValueError(f"Missing required settings for MCP servers: {', '.join(missing)}")

    configs: Dict[str, Dict[str, Any]] = {
        "github": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_personal_access_token
            },
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                settings.repo_path,
            ],
        },
        "postgres": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "postgres-mcp-server"],
            "env": {
                "DATABASE_URL": settings.database_url
            },
        },
    }

    logging.getLogger(__name__).info("Built MCP server configs: %s", list(configs.keys()))
    return configs
