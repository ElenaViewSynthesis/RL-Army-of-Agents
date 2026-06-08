"""
Persist and reload knowledge graphs as JSON (node-link format).
Output directory: data/graphs/
"""

import json
import logging
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

log = logging.getLogger(__name__)

GRAPHS_DIR = Path("data/graphs")
RAW_DIR = Path("data/raw")


def _ensure_dirs():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_graph(G: nx.DiGraph, issuer_slug: str) -> Path:
    _ensure_dirs()
    path = GRAPHS_DIR / f"{issuer_slug}.json"
    data = json_graph.node_link_data(G)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    log.info("Saved graph → %s", path)
    return path


def load_graph(issuer_slug: str) -> nx.DiGraph:
    path = GRAPHS_DIR / f"{issuer_slug}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(data, directed=True)


def save_raw(data: dict | list, name: str) -> Path:
    """Persist raw scraped data for debugging/replay."""
    _ensure_dirs()
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def list_saved_graphs() -> list[str]:
    _ensure_dirs()
    return [p.stem for p in sorted(GRAPHS_DIR.glob("*.json"))]


def save_combined_index(summaries: list[dict]) -> Path:
    """Write a top-level index.json with summary stats for all issuers."""
    _ensure_dirs()
    path = GRAPHS_DIR / "index.json"
    path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    log.info("Saved combined index → %s", path)
    return path
