import pytest

pytest.importorskip("langchain_mcp_adapters")

from agent.mcp_client.client import get_mcp_tools_with_namespaces


@pytest.mark.integration
@pytest.mark.asyncio
async def test_discovers_expected_namespaces() -> None:
    tools, by_namespace = await get_mcp_tools_with_namespaces()

    assert tools
    assert {"fs", "git", "ast", "test", "deps", "ci"}.issubset(by_namespace)
    assert any(tool.name == "read_file" for tool in by_namespace["fs"])
    assert any(tool.name == "git_status" for tool in by_namespace["git"])
