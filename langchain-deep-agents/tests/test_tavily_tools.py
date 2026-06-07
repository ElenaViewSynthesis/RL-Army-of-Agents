from app.config.settings import Settings
from app.tools.tavily_tools import build_tavily_tools


def test_build_tavily_tools_skips_missing_key():
    tools = build_tavily_tools(Settings(tavily_api_key=None))

    assert tools == []


def test_build_tavily_tools_creates_extract_crawl_and_map():
    tools = build_tavily_tools(Settings(tavily_api_key="tvly-test"))

    assert [tool.name for tool in tools] == [
        "tavily_extract",
        "tavily_crawl",
        "tavily_map",
    ]
