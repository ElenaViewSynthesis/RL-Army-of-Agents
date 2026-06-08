"""
Builds a NetworkX DiGraph (knowledge graph) for a single ETF issuer.

Graph structure
───────────────
  Issuer node  (id = issuer.slug, e.g. "vanguard")
      │
      └── ETF node  (id = etf.symbol, e.g. "VOO")
              relationship: "issues"

Node attributes
───────────────
  Issuer: type, name, slug, acronym, url, aum_billions
  ETF:    type, name, asset_class, aum_millions, fund_flow_pct,
          avg_volume, price, one_year_return_pct

Edge attributes
───────────────
  relationship: "issues"
  asset_class:  ETF asset class (always "Equity" in this pipeline)
"""

import logging

import networkx as nx

from models import Issuer, ETF

log = logging.getLogger(__name__)


def build_issuer_graph(issuer: Issuer) -> nx.DiGraph:
    """
    Return a directed knowledge graph for *issuer* containing only equity ETFs.
    The root node id is the issuer slug; each equity ETF is a child node.
    """
    G = nx.DiGraph(
        issuer_name=issuer.name,
        issuer_slug=issuer.slug,
        acronym=issuer.acronym,
    )

    # Root node
    G.add_node(issuer.slug, **issuer.to_node_attrs())

    equity_etfs = issuer.equity_etfs
    log.info("Building graph for %s with %d equity ETFs", issuer.name, len(equity_etfs))

    for etf in equity_etfs:
        G.add_node(etf.symbol, **etf.to_node_attrs())
        G.add_edge(
            issuer.slug,
            etf.symbol,
            relationship="issues",
            asset_class=etf.asset_class,
        )

    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Return basic statistics for a graph."""
    issuer_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "issuer"]
    etf_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "etf"]

    aum_vals = [
        G.nodes[n].get("aum_millions")
        for n in etf_nodes
        if G.nodes[n].get("aum_millions") is not None
    ]

    return {
        "issuer": issuer_nodes[0] if issuer_nodes else None,
        "equity_etf_count": len(etf_nodes),
        "total_aum_millions": round(sum(aum_vals), 2) if aum_vals else None,
        "largest_etf_by_aum": (
            max(etf_nodes, key=lambda n: G.nodes[n].get("aum_millions") or 0)
            if etf_nodes else None
        ),
    }


def merge_graphs(graphs: list[nx.DiGraph]) -> nx.DiGraph:
    """
    Combine multiple issuer graphs into one unified knowledge graph.
    Shared asset-class category nodes are added as intermediaries when present.
    """
    combined = nx.DiGraph()
    for G in graphs:
        combined.update(G)
    return combined
