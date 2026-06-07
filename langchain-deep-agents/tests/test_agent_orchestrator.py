import pytest

from app.config.settings import Settings
from app.services.agent_orchestrator import AgentOrchestrator


@pytest.mark.asyncio
async def test_initialize_adds_tavily_tools_to_research_agent(monkeypatch):
    captured = {}

    class FakeMCPClient:
        def get_tools(self):
            return ["mcp-tool"]

    async def create_mcp_client(settings):
        return FakeMCPClient()

    def build_tavily_tools(settings):
        return ["tavily-tool"]

    def build_research_agent(model, tools):
        captured["research_tools"] = tools
        return "research-agent"

    def build_code_agent(model, tools):
        captured["code_tools"] = tools
        return "code-agent"

    def build_db_agent(model, tools):
        captured["db_tools"] = tools
        return "db-agent"

    def build_main_agent(model, tools):
        captured["main_tools"] = tools
        return "main-agent"

    monkeypatch.setattr("app.mcp.client_factory.create_mcp_client", create_mcp_client)
    monkeypatch.setattr("app.tools.tavily_tools.build_tavily_tools", build_tavily_tools)
    monkeypatch.setattr("app.agents.research_agent.build_research_agent", build_research_agent)
    monkeypatch.setattr("app.agents.code_agent.build_code_agent", build_code_agent)
    monkeypatch.setattr("app.agents.db_agent.build_db_agent", build_db_agent)
    monkeypatch.setattr("app.agents.main_agent.build_main_agent", build_main_agent)
    monkeypatch.setattr(
        "app.tools.subagent_tools.build_research_tool",
        lambda agent: "research-tool",
    )
    monkeypatch.setattr(
        "app.tools.subagent_tools.build_codebase_tool",
        lambda agent: "code-tool",
    )
    monkeypatch.setattr(
        "app.tools.subagent_tools.build_database_tool",
        lambda agent: "db-tool",
    )

    orchestrator = AgentOrchestrator(Settings(tavily_api_key="tvly-test"))
    await orchestrator.initialize()

    assert orchestrator.tavily_tools == ["tavily-tool"]
    assert captured["research_tools"] == ["mcp-tool", "tavily-tool"]
    assert captured["code_tools"] == ["mcp-tool"]
    assert captured["db_tools"] == ["mcp-tool"]
    assert captured["main_tools"] == ["mcp-tool", "research-tool", "code-tool", "db-tool"]
