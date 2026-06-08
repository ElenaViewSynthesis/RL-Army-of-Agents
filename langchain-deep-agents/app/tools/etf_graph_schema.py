"""ETF graph schema, payload helpers, and Neo4j connection utilities."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from app.config.settings import Settings
from app.tools.neo4j_graph_tools import graph_documents_from_payload


ETFDB_MARKET_CAP_URL = "https://etfdb.com/compare/market-cap/"

ETF_TABLE_LABEL = "ETFTable"
ETF_LABEL = "ETF"
HAS_ETF_RELATIONSHIP = "HAS_ETF"

ETF_GRAPH_SCHEMA: dict[str, Any] = {
    "nodes": {
        ETF_TABLE_LABEL: {
            "properties": ["name", "source_url", "retrieved_at"],
            "description": "A source table containing ETF market-cap ranking rows.",
        },
        ETF_LABEL: {
            "properties": [
                "symbol",
                "name",
                "aum",
                "avg_daily_share_volume_3mo",
                "etfdb_url",
                "source",
            ],
            "description": "An ETF row from the ETFDB market-cap comparison table.",
        },
    },
    "relationships": {
        HAS_ETF_RELATIONSHIP: {
            "source": ETF_TABLE_LABEL,
            "target": ETF_LABEL,
            "description": "Connects an ETFDB market-cap table snapshot to each ETF in it.",
        }
    },
}

ETFDB_MARKET_CAP_ETFS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "VOO",
        "name": "Vanguard S&P 500 ETF",
        "aum": 1_003_510_000,
        "avg_daily_share_volume_3mo": 8_762_368,
    },
    {
        "symbol": "IVV",
        "name": "iShares Core S&P 500 ETF",
        "aum": 851_402_000,
        "avg_daily_share_volume_3mo": 8_100_509,
    },
    {
        "symbol": "SPY",
        "name": "State Street SPDR S&P 500 ETF",
        "aum": 786_553_000,
        "avg_daily_share_volume_3mo": 67_128_516,
    },
    {
        "symbol": "VTI",
        "name": "Vanguard Total Stock Market ETF",
        "aum": 661_247_000,
        "avg_daily_share_volume_3mo": 4_319_158,
    },
    {
        "symbol": "QQQ",
        "name": "Invesco QQQ Trust Series I",
        "aum": 495_519_000,
        "avg_daily_share_volume_3mo": 50_899_555,
    },
    {
        "symbol": "VEA",
        "name": "Vanguard FTSE Developed Markets ETF",
        "aum": 231_012_000,
        "avg_daily_share_volume_3mo": 14_523_029,
    },
    {
        "symbol": "VUG",
        "name": "Vanguard Growth ETF",
        "aum": 230_150_000,
        "avg_daily_share_volume_3mo": 8_334_692,
    },
    {
        "symbol": "IEFA",
        "name": "iShares Core MSCI EAFE ETF",
        "aum": 184_960_000,
        "avg_daily_share_volume_3mo": 12_903_533,
    },
    {
        "symbol": "VTV",
        "name": "Vanguard Value ETF",
        "aum": 180_296_000,
        "avg_daily_share_volume_3mo": 3_573_129,
    },
    {
        "symbol": "IEMG",
        "name": "iShares Core MSCI Emerging Markets ETF",
        "aum": 165_959_000,
        "avg_daily_share_volume_3mo": 14_984_369,
    },
    {
        "symbol": "MGK",
        "name": "Vanguard Mega Cap Growth ETF",
        "aum": 34_583_400,
        "avg_daily_share_volume_3mo": 1_947_599,
    },
    {
        "symbol": "ACWI",
        "name": "iShares MSCI ACWI ETF",
        "aum": 32_936_600,
        "avg_daily_share_volume_3mo": 4_560_149,
    },
    {
        "symbol": "IUSG",
        "name": "iShares Core S&P U.S. Growth ETF",
        "aum": 32_485_400,
        "avg_daily_share_volume_3mo": 870_540,
    },
    {
        "symbol": "EEM",
        "name": "iShares MSCI Emerging Markets ETF",
        "aum": 31_221_500,
        "avg_daily_share_volume_3mo": 35_240_820,
    },
    {
        "symbol": "SOXL",
        "name": "Direxion Daily Semiconductor Bull 3X ETF",
        "aum": 30_552_300,
        "avg_daily_share_volume_3mo": 76_649_695,
    },
    {
        "symbol": "XLI",
        "name": "State Street Industrial Select Sector SPDR ETF",
        "aum": 30_445_400,
        "avg_daily_share_volume_3mo": 10_844_672,
    },
    {
        "symbol": "IDEV",
        "name": "iShares Core MSCI International Developed Markets ETF",
        "aum": 30_387_500,
        "avg_daily_share_volume_3mo": 1_611_558,
    },
    {
        "symbol": "VXF",
        "name": "Vanguard Extended Market ETF",
        "aum": 30_216_800,
        "avg_daily_share_volume_3mo": 430_923,
    },
    {
        "symbol": "GLDM",
        "name": "SPDR Gold Minishares Trust",
        "aum": 30_159_400,
        "avg_daily_share_volume_3mo": 5_190_463,
    },
    {
        "symbol": "VGK",
        "name": "Vanguard FTSE Europe ETF",
        "aum": 29_769_000,
        "avg_daily_share_volume_3mo": 4_053_614,
    },
    {
        "symbol": "SCHB",
        "name": "Schwab U.S. Broad Market ETF",
        "aum": 43_208_800,
        "avg_daily_share_volume_3mo": 10_407_433,
    },
    {
        "symbol": "GOVT",
        "name": "iShares U.S. Treasury Bond ETF",
        "aum": 42_510_500,
        "avg_daily_share_volume_3mo": 12_602_614,
    },
    {
        "symbol": "IUSB",
        "name": "iShares Core Total USD Bond Market ETF",
        "aum": 41_520_900,
        "avg_daily_share_volume_3mo": 5_789_132,
    },
    {
        "symbol": "SOXX",
        "name": "iShares Semiconductor ETF",
        "aum": 41_348_500,
        "avg_daily_share_volume_3mo": 7_732_725,
    },
    {
        "symbol": "VGIT",
        "name": "Vanguard Intermediate-Term Treasury ETF",
        "aum": 41_052_500,
        "avg_daily_share_volume_3mo": 3_312_097,
    },
    {
        "symbol": "TLT",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "aum": 40_945_000,
        "avg_daily_share_volume_3mo": 31_060_699,
    },
    {
        "symbol": "SPDW",
        "name": "State Street SPDR Portfolio Developed World ex-US ETF",
        "aum": 40_851_000,
        "avg_daily_share_volume_3mo": 4_853_139,
    },
    {
        "symbol": "DGRO",
        "name": "iShares Core Dividend Growth ETF",
        "aum": 40_536_400,
        "avg_daily_share_volume_3mo": 2_067_146,
    },
    {
        "symbol": "TQQQ",
        "name": "ProShares UltraPro QQQ",
        "aum": 40_277_800,
        "avg_daily_share_volume_3mo": 89_067_180,
    },
    {
        "symbol": "XLE",
        "name": "State Street Energy Select Sector SPDR ETF",
        "aum": 40_067_700,
        "avg_daily_share_volume_3mo": 49_043_742,
    },
)


def _etf_node_id(symbol: str) -> str:
    return f"etf:{symbol.upper()}"


def _etfdb_url(symbol: str) -> str:
    return f"https://etfdb.com/etf/{symbol.upper()}/"


def _load_settings() -> Settings:
    load_dotenv(".env")
    load_dotenv(Path("..") / ".env", override=False)
    parent_env = Path("..") / ".env"
    if parent_env.exists():
        return Settings(_env_file=parent_env)
    return Settings()


def build_etf_graph_payload(
    *,
    retrieved_at: str,
    etfs: Sequence[Mapping[str, Any]] = ETFDB_MARKET_CAP_ETFS,
    source_url: str = ETFDB_MARKET_CAP_URL,
) -> dict[str, Any]:
    """Build a GraphDocument-compatible JSON payload for ETFDB market-cap rows."""
    table_id = f"etf_table:{source_url}"
    nodes: list[dict[str, Any]] = [
        {
            "id": table_id,
            "type": ETF_TABLE_LABEL,
            "properties": {
                "name": "ETFDB ETF Market Cap Comparison",
                "source_url": source_url,
                "retrieved_at": retrieved_at,
            },
        }
    ]
    relationships: list[dict[str, Any]] = []

    for etf in etfs:
        symbol = str(etf["symbol"]).upper()
        nodes.append(
            {
                "id": _etf_node_id(symbol),
                "type": ETF_LABEL,
                "properties": {
                    "symbol": symbol,
                    "name": etf["name"],
                    "aum": etf["aum"],
                    "avg_daily_share_volume_3mo": etf["avg_daily_share_volume_3mo"],
                    "etfdb_url": etf.get("etfdb_url", _etfdb_url(symbol)),
                    "source": etf.get("source", "etfdb_market_cap"),
                },
            }
        )
        relationships.append(
            {
                "source": table_id,
                "target": _etf_node_id(symbol),
                "type": HAS_ETF_RELATIONSHIP,
                "properties": {"source_url": source_url, "retrieved_at": retrieved_at},
            }
        )

    return {
        "documents": [
            {
                "source": {
                    "url": source_url,
                    "title": "ETFDB ETF Market Cap Comparison",
                    "summary": "ETF market-cap rows scraped from ETFDB.",
                    "retrieved_at": retrieved_at,
                },
                "nodes": nodes,
                "relationships": relationships,
            }
        ]
    }


def connect_neo4j_graph(settings: Settings | None = None):
    """Create a Neo4jGraph connection using application settings."""
    from langchain_neo4j import Neo4jGraph

    settings = settings or _load_settings()
    kwargs: dict[str, Any] = {
        "url": settings.neo4j_uri,
        "username": settings.neo4j_username,
        "password": settings.neo4j_password,
        "refresh_schema": False,
    }
    if settings.neo4j_database:
        kwargs["database"] = settings.neo4j_database
    return Neo4jGraph(**kwargs)


def store_etf_graph_schema(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Persist schema metadata nodes for ETFTable, ETF, and HAS_ETF."""
    graph = connect_neo4j_graph(settings)
    return graph.query(
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
            "table_label": ETF_TABLE_LABEL,
            "table_properties": ETF_GRAPH_SCHEMA["nodes"][ETF_TABLE_LABEL]["properties"],
            "etf_label": ETF_LABEL,
            "etf_properties": ETF_GRAPH_SCHEMA["nodes"][ETF_LABEL]["properties"],
            "relationship_type": HAS_ETF_RELATIONSHIP,
        },
    )


def store_etf_market_cap_graph(
    *,
    settings: Settings | None = None,
    retrieved_at: str | None = None,
    etfs: Sequence[Mapping[str, Any]] = ETFDB_MARKET_CAP_ETFS,
    source_url: str = ETFDB_MARKET_CAP_URL,
) -> dict[str, int]:
    """Connect to Neo4j and persist ETFDB market-cap ETFTable/ETF graph data."""
    retrieved_at = retrieved_at or datetime.now(UTC).isoformat()
    payload = build_etf_graph_payload(
        retrieved_at=retrieved_at,
        etfs=etfs,
        source_url=source_url,
    )
    graph_documents = graph_documents_from_payload(payload)
    graph = connect_neo4j_graph(settings)
    graph.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=True,
    )

    graph_document = graph_documents[0]
    return {
        "graph_documents": len(graph_documents),
        "nodes": len(graph_document.nodes),
        "relationships": len(graph_document.relationships),
    }


def query_etf_market_cap_graph(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Return ETF rows attached to the ETFDB market-cap table."""
    graph = connect_neo4j_graph(settings)
    return graph.query(
        """
        MATCH (table:ETFTable)-[:HAS_ETF]->(etf:ETF)
        RETURN table.name AS table_name,
               etf.symbol AS symbol,
               etf.name AS name,
               etf.aum AS aum,
               etf.avg_daily_share_volume_3mo AS avg_daily_share_volume_3mo
        ORDER BY etf.aum DESC
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store or query the ETFDB market-cap graph in Neo4j."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="store-market-cap",
        choices=("store-schema", "store-market-cap", "query"),
        help=(
            "store-market-cap loads ETFTable/ETF rows, store-schema stores schema "
            "metadata, and query prints loaded ETF rows."
        ),
    )
    args = parser.parse_args()

    if args.command == "store-schema":
        result: Any = store_etf_graph_schema()
    elif args.command == "query":
        result = query_etf_market_cap_graph()
    else:
        result = store_etf_market_cap_graph()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
