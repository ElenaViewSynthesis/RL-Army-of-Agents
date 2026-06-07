"""Tavily tool collection for web research specialists."""
from __future__ import annotations

import logging
from typing import Any

from app.config.settings import Settings


logger = logging.getLogger(__name__)

_PLACEHOLDER_KEYS = {"", "your-api-key", "changeme", "change-me"}


def build_tavily_tools(settings: Settings) -> list[Any]:
    """Build Tavily extract, crawl, and map tools when configured."""
    api_key = (settings.tavily_api_key or "").strip()
    if api_key.lower() in _PLACEHOLDER_KEYS:
        logger.info("TAVILY_API_KEY is not configured; skipping Tavily tools")
        return []

    try:
        from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
        from langchain_tavily._utilities import (
            TavilyCrawlAPIWrapper,
            TavilyExtractAPIWrapper,
            TavilyMapAPIWrapper,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("langchain-tavily is not available; skipping Tavily tools: %s", exc)
        return []

    try:
        return [
            TavilyExtract(
                apiwrapper=TavilyExtractAPIWrapper(tavily_api_key=api_key),
            ),
            TavilyCrawl(
                api_wrapper=TavilyCrawlAPIWrapper(tavily_api_key=api_key),
            ),
            TavilyMap(
                api_wrapper=TavilyMapAPIWrapper(tavily_api_key=api_key),
            ),
        ]
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize Tavily tools: %s", exc)
        return []
