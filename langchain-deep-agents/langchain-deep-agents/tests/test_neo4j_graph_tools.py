import pytest

from app.tools.neo4j_graph_tools import (
    _validate_read_only_cypher,
    graph_documents_from_payload,
)


def test_graph_documents_from_payload_builds_nodes_relationships_and_source():
    payload = {
        "documents": [
            {
                "source": {
                    "url": "https://example.com/page",
                    "title": "Example",
                    "content": "Example page content",
                },
                "nodes": [
                    {
                        "id": "page:https://example.com/page",
                        "type": "WebPage",
                        "properties": {"url": "https://example.com/page"},
                    },
                    {
                        "id": "entity:Neo4j",
                        "type": "Technology",
                        "properties": {"name": "Neo4j"},
                    },
                ],
                "relationships": [
                    {
                        "source": "page:https://example.com/page",
                        "target": "entity:Neo4j",
                        "type": "mentions technology",
                        "properties": {"confidence": 0.9},
                    }
                ],
            }
        ]
    }

    graph_documents = graph_documents_from_payload(payload)

    assert len(graph_documents) == 1
    graph_document = graph_documents[0]
    assert len(graph_document.nodes) == 2
    assert graph_document.nodes[0].type == "WebPage"
    assert len(graph_document.relationships) == 1
    assert graph_document.relationships[0].type == "MENTIONS_TECHNOLOGY"
    assert graph_document.source.page_content == "Example page content"
    assert graph_document.source.metadata["url"] == "https://example.com/page"


def test_graph_documents_from_payload_rejects_unknown_relationship_nodes():
    payload = {
        "nodes": [{"id": "known", "type": "Entity"}],
        "relationships": [{"source": "known", "target": "missing", "type": "RELATED_TO"}],
    }

    with pytest.raises(ValueError, match="unknown nodes"):
        graph_documents_from_payload(payload)


def test_validate_read_only_cypher_allows_match_and_blocks_writes():
    assert _validate_read_only_cypher("MATCH (n) RETURN n LIMIT 5").startswith("MATCH")

    with pytest.raises(ValueError, match="read-only"):
        _validate_read_only_cypher("MATCH (n) DETACH DELETE n")
