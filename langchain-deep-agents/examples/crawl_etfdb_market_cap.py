"""Example: scrape ETFDB market-cap table content with Tavily Crawl.

Run from the repository root:
uv --cache-dir /tmp/uv-cache --project langchain-deep-agents run python \
    langchain-deep-agents/examples/crawl_etfdb_market_cap.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_tavily import TavilyCrawl


SOURCE_URL = "https://etfdb.com/compare/market-cap/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "etfdb_market_cap_content.md"


def _extract_pages(response: Any) -> list[dict[str, Any]]:
    """Normalize Tavily Crawl response shapes across SDK versions."""
    if isinstance(response, dict):
        results = response.get("results") or response.get("pages") or []
        return [page for page in results if isinstance(page, dict)]

    if isinstance(response, list):
        return [page for page in response if isinstance(page, dict)]

    return []


def _page_content(page: dict[str, Any]) -> str:
    for key in ("raw_content", "content", "markdown", "text"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def crawl_etfdb_market_cap() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / ".env", override=False)

    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Set TAVILY_API_KEY in your environment or .env file.")

    crawl = TavilyCrawl(
        max_depth=1,
        max_breadth=1,
        limit=1,
        extract_depth="advanced",
        format="markdown",
        include_images=False,
        allow_external=False,
    )

    response = crawl.invoke(
        {
            "url": SOURCE_URL,
            "instructions": (
                "Extract the ETF market-cap table, including Symbol, Name, AUM, "
                "and average daily share volume."
            ),
        }
    )

    pages = _extract_pages(response)
    if not pages:
        raise RuntimeError(f"Tavily Crawl returned no pages: {response!r}")

    content = _page_content(pages[0])
    if not content:
        raise RuntimeError(f"Tavily Crawl returned no page content: {pages[0]!r}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content + "\n", encoding="utf-8")

    return content


if __name__ == "__main__":
    scraped_content = crawl_etfdb_market_cap()
    print(f"Scraped {len(scraped_content):,} characters from {SOURCE_URL}")
    print(f"Wrote content to {OUTPUT_PATH}")
    print(scraped_content[:1000])
