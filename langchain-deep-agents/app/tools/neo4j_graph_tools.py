"""Neo4j graph storage and read-only query tools."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config.settings import Settings
from app.tools.subagent_tools import _safe_tool_decorator


logger = logging.getLogger(__name__)

_PLACEHOLDER_VALUES = {"", "changeme", "change-me", "password", "your-password"}
_WRITE_CYPHER_PATTERN = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|load\s+csv|call\s+apoc)\b",
    re.IGNORECASE,
)


def _neo4j_configured(settings: Settings) -> bool:
    return all(
        [
            (settings.neo4j_uri or "").strip(),
            (settings.neo4j_username or "").strip(),
            (settings.neo4j_password or "").strip().lower() not in _PLACEHOLDER_VALUES,
        ]
    )


def _create_neo4j_graph(settings: Settings):
    from langchain_neo4j import Neo4jGraph

    kwargs: dict[str, Any] = {
        "url": settings.neo4j_uri,
        "username": settings.neo4j_username,
        "password": settings.neo4j_password,
        "refresh_schema": False,
    }
    if settings.neo4j_database:
        kwargs["database"] = settings.neo4j_database
    return Neo4jGraph(**kwargs)


def _coerce_payload(payload_json: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload_json, str):
        data = json.loads(payload_json)
    else:
        data = payload_json
    if not isinstance(data, dict):
        raise ValueError("Graph payload must be a JSON object")
    return data


def _normalise_relationship_type(value: str) -> str:
    normalised = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
    return normalised or "RELATED_TO"


def graph_documents_from_payload(payload_json: str | dict[str, Any]):
    """Convert JSON payload into Neo4j GraphDocument objects."""
    from langchain_core.documents import Document
    from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

    payload = _coerce_payload(payload_json)
    raw_documents = payload.get("documents")
    if raw_documents is None:
        raw_documents = [payload]
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("Graph payload must contain at least one document")

    graph_documents = []
    for index, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, dict):
            raise ValueError(f"Document {index} must be a JSON object")

        nodes_by_id: dict[str, Node] = {}
        for raw_node in raw_document.get("nodes", []):
            if not isinstance(raw_node, dict):
                raise ValueError(f"Document {index} contains a non-object node")
            node_id = str(raw_node.get("id", "")).strip()
            if not node_id:
                raise ValueError(f"Document {index} contains a node without an id")
            node_type = str(raw_node.get("type") or "Entity").strip()
            properties = raw_node.get("properties") or {}
            if not isinstance(properties, dict):
                raise ValueError(f"Node {node_id} properties must be an object")
            nodes_by_id[node_id] = Node(id=node_id, type=node_type, properties=properties)

        relationships = []
        for raw_relationship in raw_document.get("relationships", []):
            if not isinstance(raw_relationship, dict):
                raise ValueError(f"Document {index} contains a non-object relationship")
            source_id = str(raw_relationship.get("source", "")).strip()
            target_id = str(raw_relationship.get("target", "")).strip()
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise ValueError(
                    f"Relationship {source_id}->{target_id} references unknown nodes"
                )
            rel_type = _normalise_relationship_type(
                str(raw_relationship.get("type") or "RELATED_TO")
            )
            properties = raw_relationship.get("properties") or {}
            if not isinstance(properties, dict):
                raise ValueError(
                    f"Relationship {source_id}->{target_id} properties must be an object"
                )
            relationships.append(
                Relationship(
                    source=nodes_by_id[source_id],
                    target=nodes_by_id[target_id],
                    type=rel_type,
                    properties=properties,
                )
            )

        source = raw_document.get("source") or {}
        if not isinstance(source, dict):
            raise ValueError(f"Document {index} source must be an object")
        page_content = str(source.get("content") or source.get("summary") or "")
        metadata = {key: value for key, value in source.items() if key != "content"}

        graph_documents.append(
            GraphDocument(
                nodes=list(nodes_by_id.values()),
                relationships=relationships,
                source=Document(page_content=page_content, metadata=metadata),
            )
        )

    return graph_documents


def _validate_read_only_cypher(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Cypher query is empty")
    if _WRITE_CYPHER_PATTERN.search(cleaned):
        raise ValueError("Only read-only Cypher queries are allowed")
    allowed_prefixes = ("match ", "with ", "return ", "call db.", "call dbms.")
    if not cleaned.lower().startswith(allowed_prefixes):
        raise ValueError("Cypher query must start with MATCH, WITH, RETURN, CALL db., or CALL dbms.")
    return cleaned


def build_neo4j_storage_tools(settings: Settings) -> list[Any]:
    """Build graph storage tools when Neo4j is configured."""
    if not _neo4j_configured(settings):
        logger.info("Neo4j settings are not configured; skipping graph storage tools")
        return []

    decorator = _safe_tool_decorator(
        "store_web_graph_documents",
        (
            "Persist JSON graph documents to Neo4j. Input must be JSON with a "
            "'documents' array containing nodes, relationships, and source metadata."
        ),
    )

    @decorator
    def store_web_graph_documents(graph_documents_json: str) -> str:
        try:
            graph_documents = graph_documents_from_payload(graph_documents_json)
            graph = _create_neo4j_graph(settings)
            graph.add_graph_documents(
                graph_documents,
                include_source=True,
                baseEntityLabel=True,
            )
        except Exception as exc:
            logger.exception("Failed to store graph documents: %s", exc)
            return f"error: {exc}"
        return f"stored {len(graph_documents)} graph document(s) in Neo4j"

    return [store_web_graph_documents]


def build_neo4j_query_tools(settings: Settings) -> list[Any]:
    """Build read-only graph query tools when Neo4j is configured."""
    if not _neo4j_configured(settings):
        logger.info("Neo4j settings are not configured; skipping graph query tools")
        return []

    decorator = _safe_tool_decorator(
        "query_web_knowledge_graph",
        "Run a read-only Cypher query against the Neo4j web knowledge graph.",
    )

    @decorator
    def query_web_knowledge_graph(cypher: str) -> str:
        try:
            query = _validate_read_only_cypher(cypher)
            graph = _create_neo4j_graph(settings)
            rows = graph.query(query)
        except Exception as exc:
            logger.exception("Failed to query Neo4j graph: %s", exc)
            return f"error: {exc}"
        return json.dumps(rows, default=str)

    return [query_web_knowledge_graph]


def build_neo4j_tools(settings: Settings) -> list[Any]:
    """Build all Neo4j graph tools."""
    return [*build_neo4j_storage_tools(settings), *build_neo4j_query_tools(settings)]
