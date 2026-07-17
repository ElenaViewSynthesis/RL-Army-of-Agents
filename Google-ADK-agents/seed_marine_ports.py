#!/usr/bin/env python3
"""Seed marine fuel (bunker) ports into the TimescaleDB `marine_ports` columnstore.

Reference/dimension data (ports change rarely), stored as a **time-series of
snapshots**: each run stamps a `snapshot_ts`, so the columnstore hypertable keeps
a compressible history of how port capabilities (fuel grades, hours) evolve.

Source: OilPrice `GET /v1/marine-ports` (free). One request per run.

    uv run --extra timescale python seed_marine_ports.py
    uv run --extra timescale python seed_marine_ports.py --region Asia --major

Self-contained (httpx + psycopg via tiger_client). Reads keys from
finance_coordinator/.env.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

OILPRICE_BASE = "https://api.oilpriceapi.com/v1"


def fetch_ports(key: str, region: str, country: str, major_only: bool) -> list[dict]:
    params: dict = {}
    if region:
        params["region"] = region
    if country:
        params["country"] = country
    if major_only:
        params["major_ports"] = "true"
    r = httpx.get(f"{OILPRICE_BASE}/marine-ports", params=params,
                  headers={"Authorization": f"Token {key}"}, timeout=20)
    r.raise_for_status()
    return ((r.json().get("data") or {}).get("ports") or [])


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / "finance_coordinator" / ".env")

    ap = argparse.ArgumentParser(description="Seed marine_ports (columnstore) in TimescaleDB.")
    ap.add_argument("--region", default="", help="Asia / Europe / Americas / Middle East")
    ap.add_argument("--country", default="", help="country code filter")
    ap.add_argument("--major", action="store_true", help="major bunkering hubs only")
    args = ap.parse_args()

    key = os.getenv("OILPRICE_API_KEY")
    if not key:
        print("Error: OILPRICE_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1

    from a2a_finance import tiger_client
    if not tiger_client.enabled():
        print("Error: TimescaleDB not configured (TIGER_DATABASE_* + timescale extra).",
              file=sys.stderr)
        return 1

    ports = fetch_ports(key, args.region, args.country, args.major)
    snapshot = datetime.now(timezone.utc)
    rows = []
    for p in ports:
        coords = p.get("coordinates") or {}
        rows.append({
            "snapshot_ts": snapshot,
            "code": p.get("code"),
            "name": p.get("name"),
            "country": p.get("country"),
            "region": p.get("region"),
            "major_port": p.get("major_port"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "fuel_services": p.get("fuel_services"),
            "trading_hours": p.get("trading_hours"),
        })

    written = tiger_client.save_marine_ports(rows)
    print(f"▸ fetched {len(ports)} port(s); upserted {written} into marine_ports "
          f"(snapshot {snapshot.isoformat()}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
