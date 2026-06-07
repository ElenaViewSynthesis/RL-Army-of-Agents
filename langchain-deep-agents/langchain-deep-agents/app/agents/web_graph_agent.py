"""Build a web-to-graph subagent for Tavily ingestion into Neo4j."""
from __future__ import annotations

import logging
from typing import Any, List


def build_web_graph_agent(model: str, tools: List[Any]):
    system_message = (
        "You are a web knowledge graph ingestion subagent.\n\n"
        "Your job is to turn web evidence into a queryable Neo4j knowledge graph:\n"
        "1. Search the web with TavilySearch.\n"
        "2. Fetch useful pages with TavilyExtract, and use TavilyCrawl or TavilyMap "
        "when a bounded site crawl or URL map is needed.\n"
        "3. Convert useful results into graph entities and relationships.\n"
        "4. Persist those entities using store_web_graph_documents.\n"
        "5. Query the graph with query_web_knowledge_graph when you need to verify "
        "what is already stored.\n\n"
        "Use JSON graph documents with this shape when storing:\n"
        "{'documents':[{'source':{'url':'...', 'title':'...', 'content':'...'}, "
        "'nodes':[{'id':'...', 'type':'Entity', 'properties':{...}}], "
        "'relationships':[{'source':'...', 'target':'...', 'type':'MENTIONS', "
        "'properties':{...}}]}]}.\n\n"
        "Prefer stable IDs derived from URLs and canonical entity names. Include source URLs, "
        "titles, snippets, extraction timestamps when available, and confidence/evidence "
        "properties on relationships. Do not invent facts that are not supported by the fetched pages.\n"
        "Return a concise ingestion report listing searched queries, fetched URLs, stored node "
        "and relationship counts, and useful Cypher queries the main agent can run later."
    )

    try:
        from langchain.agents import create_agent
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning(
            "langchain not available, returning stub web graph agent: %s", exc
        )

        class _Stub:
            def __init__(self, system_message: str):
                self.system_message = system_message

            def invoke(self, payload: dict):
                return {"messages": [{"role": "assistant", "content": "stub web graph result"}]}

            async def ainvoke(self, payload: dict):
                return self.invoke(payload)

        return _Stub(system_message)

    agent = create_agent(model, tools)

    class AgentWrapper:
        def __init__(self, agent: Any, system_message: str):
            self._agent = agent
            self._system_message = system_message

        def invoke(self, payload: dict):
            messages = payload.get("messages", [])
            messages = [{"role": "system", "content": self._system_message}] + messages
            payload = {"messages": messages}
            if hasattr(self._agent, "invoke"):
                return self._agent.invoke(payload)
            return {"messages": [{"role": "assistant", "content": "agent missing invoke"}]}

        async def ainvoke(self, payload: dict):
            messages = payload.get("messages", [])
            messages = [{"role": "system", "content": self._system_message}] + messages
            payload = {"messages": messages}
            if hasattr(self._agent, "ainvoke"):
                return await self._agent.ainvoke(payload)
            if hasattr(self._agent, "invoke"):
                return self._agent.invoke(payload)
            return {"messages": [{"role": "assistant", "content": "agent missing ainvoke/invoke"}]}

    return AgentWrapper(agent, system_message)
