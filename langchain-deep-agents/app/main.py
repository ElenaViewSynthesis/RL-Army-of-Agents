"""CLI entrypoint for the agent orchestrator."""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from app.config.settings import Settings
from app.services.agent_orchestrator import AgentOrchestrator
from app.observability.logging import configure_logging


async def _run_query(query: Optional[str]) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()

    if not query:
        query = input("Enter an engineering task (single line): ")

    result = await orchestrator.run(query)
    print("\n=== Agent Report ===\n")
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the architecture agent")
    parser.add_argument("query", nargs="?", help="User query to run")
    args = parser.parse_args()

    try:
        asyncio.run(_run_query(args.query))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")


if __name__ == "__main__":
    main()
