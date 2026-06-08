"""
ETF Knowledge Graph Builder
============================
Crawls etfdb.com for ETF issuers, builds per-issuer knowledge graphs
containing equity ETFs, and persists them as JSON.

Usage
-----
  python main.py                          # all issuers, Tavily on
  python main.py --no-tavily              # direct requests fallback
  python main.py --issuers vanguard invesco blackrock   # subset
  python main.py --list                   # show saved graphs
  python main.py --show vanguard          # print graph summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scraper import fetch_issuers, fetch_issuer_etfs
from graph_builder import build_issuer_graph, graph_summary
from storage import save_graph, save_raw, save_combined_index, list_saved_graphs, load_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ETF issuer knowledge graph builder")
    p.add_argument("--no-tavily", action="store_true", help="Skip Tavily, use direct requests")
    p.add_argument("--issuers", nargs="+", metavar="SLUG",
                   help="Only process these issuer slugs (e.g. vanguard blackrock)")
    p.add_argument("--list", action="store_true", help="List saved graphs and exit")
    p.add_argument("--show", metavar="SLUG", help="Print summary of a saved graph")
    p.add_argument("--max", type=int, default=None, metavar="N",
                   help="Process at most N issuers (useful for testing)")
    return p.parse_args()


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(args):
    use_tavily = not args.no_tavily

    # ── Quick commands ────────────────────────────────────────────────────────
    if args.list:
        saved = list_saved_graphs()
        if not saved:
            print("No graphs saved yet. Run without --list to build them.")
        else:
            print(f"Saved graphs ({len(saved)}):")
            for slug in saved:
                print(f"  {slug}")
        return

    if args.show:
        try:
            G = load_graph(args.show)
            summary = graph_summary(G)
            print(json.dumps(summary, indent=2))
            # Also print all nodes
            print(f"\nNodes ({G.number_of_nodes()}):")
            for node_id, attrs in G.nodes(data=True):
                ntype = attrs.get("type", "?")
                name = attrs.get("name", node_id)
                aum = attrs.get("aum_millions")
                aum_str = f"  AUM=${aum:,.0f}M" if aum else ""
                print(f"  [{ntype}] {node_id:8s}  {name}{aum_str}")
        except FileNotFoundError:
            print(f"No saved graph for '{args.show}'. Run the pipeline first.")
        return

    # ── Full pipeline ──────────────────────────────────────────────────────────
    log.info("=== ETF Knowledge Graph Builder ===")
    log.info("Tavily: %s", "enabled" if use_tavily else "disabled (direct requests)")

    # Step 1: get issuer list
    issuers = fetch_issuers(use_tavily=use_tavily)
    if not issuers:
        log.error("No issuers found. Check connectivity or try --no-tavily.")
        sys.exit(1)

    save_raw([{"name": i.name, "slug": i.slug, "url": i.url} for i in issuers], "issuers")
    log.info("Total issuers found: %d", len(issuers))

    # Step 2: filter to requested subset
    if args.issuers:
        requested = set(args.issuers)
        issuers = [i for i in issuers if i.slug in requested]
        log.info("Filtered to %d issuer(s): %s", len(issuers), args.issuers)

    if args.max:
        issuers = issuers[: args.max]

    # Step 3: per-issuer ETF scrape + graph build
    summaries = []
    for issuer in issuers:
        try:
            equity_etfs = fetch_issuer_etfs(issuer, use_tavily=use_tavily)
            issuer.etfs = equity_etfs  # attach (already filtered to equity in scraper)

            # Ensure asset_class is set to Equity for all (scraper returns only equity)
            G = build_issuer_graph(issuer)
            path = save_graph(G, issuer.slug)

            summary = graph_summary(G)
            summary["slug"] = issuer.slug
            summary["graph_path"] = str(path)
            summaries.append(summary)

            log.info(
                "  ✓ %s  |  %d equity ETFs  |  total AUM $%sM",
                issuer.name,
                summary["equity_etf_count"],
                f"{summary['total_aum_millions']:,.0f}" if summary["total_aum_millions"] else "n/a",
            )

        except Exception as exc:
            log.error("Failed to process %s: %s", issuer.name, exc, exc_info=True)

    # Step 4: write combined index
    if summaries:
        save_combined_index(summaries)
        log.info("=== Done: %d graphs saved to data/graphs/ ===", len(summaries))
    else:
        log.warning("No graphs were built.")


if __name__ == "__main__":
    run(parse_args())
