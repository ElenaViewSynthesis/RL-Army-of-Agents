"""FastAPI graph database server backed by Neo4j and Tavily."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.config.settings import Settings
from app.tools.etf_graph_schema import (
    ETFDB_MARKET_CAP_ETFS,
    ETFDB_MARKET_CAP_URL,
    ETF_GRAPH_SCHEMA,
)
from app.tools.neo4j_graph_tools import _validate_read_only_cypher


class CypherRequest(BaseModel):
    query: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True


class TavilySearchRequest(BaseModel):
    query: str
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] | None = None
    topic: Literal["general", "news", "finance"] | None = None
    max_results: int | None = 5
    include_answer: bool | Literal["basic", "advanced"] | None = None
    include_raw_content: bool | Literal["markdown", "text"] | None = None


class TavilyExtractRequest(BaseModel):
    urls: list[str]
    extract_depth: Literal["basic", "advanced"] | None = "advanced"
    include_images: bool | None = False
    format: Literal["markdown", "text"] | None = "markdown"


class TavilyCrawlRequest(BaseModel):
    url: str
    max_depth: int | None = 1
    max_breadth: int | None = 20
    limit: int | None = 10
    extract_depth: Literal["basic", "advanced"] | None = "basic"
    format: Literal["markdown", "text"] | None = "markdown"


class TavilyMapRequest(BaseModel):
    url: str
    max_depth: int | None = 1
    max_breadth: int | None = 20
    limit: int | None = 50


class ETFIngestRequest(BaseModel):
    retrieved_at: str | None = None
    source_url: str = ETFDB_MARKET_CAP_URL


class GraphDatabaseServer:
    """Thin service wrapper around Neo4j GraphDatabase and TavilyClient."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> bool:
        self.driver.verify_connectivity()
        return True

    def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        *,
        read_only: bool = True,
    ) -> list[dict[str, Any]]:
        if read_only:
            cypher = _validate_read_only_cypher(cypher)

        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def store_etf_graph_schema(self) -> list[dict[str, Any]]:
        return self.query(
            """
            MERGE (table_schema:GraphSchema {name: $table_label})
            SET table_schema.kind = 'node',
                table_schema.properties = $table_properties
            MERGE (etf_schema:GraphSchema {name: $etf_label})
            SET etf_schema.kind = 'node',
                etf_schema.properties = $etf_properties
            MERGE (relationship_schema:GraphSchema {name: $relationship_type})
            SET relationship_schema.kind = 'relationship',
                relationship_schema.source = $table_label,
                relationship_schema.target = $etf_label
            RETURN table_schema.name AS table_label,
                   etf_schema.name AS etf_label,
                   relationship_schema.name AS relationship_type
            """,
            {
                "table_label": "ETFTable",
                "table_properties": ETF_GRAPH_SCHEMA["nodes"]["ETFTable"]["properties"],
                "etf_label": "ETF",
                "etf_properties": ETF_GRAPH_SCHEMA["nodes"]["ETF"]["properties"],
                "relationship_type": "HAS_ETF",
            },
            read_only=False,
        )

    def ingest_etf_market_cap(
        self,
        *,
        retrieved_at: str | None = None,
        source_url: str = ETFDB_MARKET_CAP_URL,
    ) -> dict[str, int]:
        retrieved_at = retrieved_at or datetime.now(UTC).isoformat()
        etfs = [
            {
                **etf,
                "symbol": str(etf["symbol"]).upper(),
                "etfdb_url": f"https://etfdb.com/etf/{str(etf['symbol']).upper()}/",
                "source": "etfdb_market_cap",
            }
            for etf in ETFDB_MARKET_CAP_ETFS
        ]

        self.query(
            """
            MERGE (table:ETFTable {source_url: $source_url})
            SET table.name = $table_name,
                table.retrieved_at = $retrieved_at
            WITH table
            UNWIND $etfs AS row
            MERGE (etf:ETF {symbol: row.symbol})
            SET etf.name = row.name,
                etf.aum = row.aum,
                etf.avg_daily_share_volume_3mo = row.avg_daily_share_volume_3mo,
                etf.etfdb_url = row.etfdb_url,
                etf.source = row.source
            MERGE (table)-[:HAS_ETF]->(etf)
            RETURN count(etf) AS etf_count
            """,
            {
                "source_url": source_url,
                "table_name": "ETFDB ETF Market Cap Comparison",
                "retrieved_at": retrieved_at,
                "etfs": etfs,
            },
            read_only=False,
        )

        return {
            "tables": 1,
            "etfs": len(etfs),
            "relationships": len(etfs),
        }

    def tavily_search(self, request: TavilySearchRequest) -> dict[str, Any]:
        return self.tavily.search(**request.model_dump(exclude_none=True))

    def tavily_extract(self, request: TavilyExtractRequest) -> dict[str, Any]:
        return self.tavily.extract(**request.model_dump(exclude_none=True))

    def tavily_crawl(self, request: TavilyCrawlRequest) -> dict[str, Any]:
        return self.tavily.crawl(**request.model_dump(exclude_none=True))

    def tavily_map(self, request: TavilyMapRequest) -> dict[str, Any]:
        return self.tavily.map(**request.model_dump(exclude_none=True))


def _load_settings() -> Settings:
    load_dotenv(".env")
    load_dotenv(Path("..") / ".env", override=False)
    parent_env = Path("..") / ".env"
    if parent_env.exists():
        return Settings(_env_file=parent_env)
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _load_settings()
    server = GraphDatabaseServer(settings)
    app.state.graph_database_server = server
    try:
        yield
    finally:
        server.close()


app = FastAPI(title="Neo4j Tavily Graph Database Server", lifespan=lifespan)


def _server() -> GraphDatabaseServer:
    server = getattr(app.state, "graph_database_server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Graph database server is not ready")
    return server


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        return {"status": "ok", "neo4j": _server().verify_connectivity()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/cypher")
def run_cypher(request: CypherRequest) -> dict[str, Any]:
    try:
        rows = _server().query(
            request.query,
            request.parameters,
            read_only=request.read_only,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"rows": rows}


@app.post("/tavily/search")
def tavily_search(request: TavilySearchRequest) -> dict[str, Any]:
    return _server().tavily_search(request)


@app.post("/tavily/extract")
def tavily_extract(request: TavilyExtractRequest) -> dict[str, Any]:
    return _server().tavily_extract(request)


@app.post("/tavily/crawl")
def tavily_crawl(request: TavilyCrawlRequest) -> dict[str, Any]:
    return _server().tavily_crawl(request)


@app.post("/tavily/map")
def tavily_map(request: TavilyMapRequest) -> dict[str, Any]:
    return _server().tavily_map(request)


@app.post("/etfdb/schema")
def store_etf_schema() -> dict[str, Any]:
    return {"rows": _server().store_etf_graph_schema()}


@app.post("/etfdb/ingest")
def ingest_etf_market_cap(request: ETFIngestRequest) -> dict[str, int]:
    return _server().ingest_etf_market_cap(
        retrieved_at=request.retrieved_at,
        source_url=request.source_url,
    )
