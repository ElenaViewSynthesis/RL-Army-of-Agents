"""Example: run a simple research task using the orchestrator."""
import asyncio

from app.config.settings import Settings
from app.services.agent_orchestrator import AgentOrchestrator
from app.observability.logging import configure_logging


async def main():
    settings = Settings()
    configure_logging(settings.log_level)
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()

    query = "Summarise the repository architecture and list missing tests."
    report = await orchestrator.run(query)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
