"""Example: run a database inspection task."""
import asyncio

from app.config.settings import Settings
from app.services.agent_orchestrator import AgentOrchestrator
from app.observability.logging import configure_logging


async def main():
    settings = Settings()
    configure_logging(settings.log_level)
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()

    query = "List database tables, relationships, and potential missing migrations."
    report = await orchestrator.run(query)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
