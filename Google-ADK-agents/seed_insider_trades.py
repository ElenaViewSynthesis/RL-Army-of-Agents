#!/usr/bin/env python3
"""Poll FMP `insider-trading/latest` and store the IMPORTANT trades in TimescaleDB.

The free feed is the latest 100 filings across all companies (no symbol filter,
no pagination — both premium). Polling it frequently and upserting builds a
local, symbol-searchable history of insider activity in the `insider_trades`
hypertable.

**Important-only filter** (signal, not noise): Form 4 + an actual open-market
buy/sell (P-Purchase / S-Sale) + a material dollar value (default >= $50k). This
drops Form 3/5, awards, option exercises, gifts, tax withholding, and tiny lots.

    uv run --extra timescale python seed_insider_trades.py
    uv run --extra timescale python seed_insider_trades.py --min-value 100000

Self-contained (httpx + psycopg via tiger_client) — no ADK import. Intended to
run frequently during US market hours (see scripts/insider_cron.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

FMP_STABLE = "https://financialmodelingprep.com/stable"
# Open-market buy/sell transaction codes worth keeping as signal.
IMPORTANT_CODES = {"P", "S"}  # P-Purchase, S-Sale


def _to_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _trade_id(rec: dict) -> str:
    key = "|".join(str(rec.get(k, "")) for k in (
        "symbol", "reportingCik", "transactionDate", "securityName",
        "transactionType", "securitiesTransacted", "price"))
    return hashlib.md5(key.encode()).hexdigest()[:20]


def _important(rec: dict, min_value: float) -> tuple[bool, float]:
    """Keep only Form-4 open-market buys/sells above a material value."""
    if str(rec.get("formType")) != "4":
        return False, 0.0
    shares = rec.get("securitiesTransacted") or 0
    price = rec.get("price") or 0
    if shares <= 0 or price <= 0:
        return False, 0.0
    code = str(rec.get("transactionType", "")).split("-", 1)[0]
    if code not in IMPORTANT_CODES:
        return False, 0.0
    value = float(shares) * float(price)
    return value >= min_value, value


def fetch_latest(key: str) -> list[dict]:
    r = httpx.get(f"{FMP_STABLE}/insider-trading/latest",
                  params={"apikey": key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / "finance_coordinator" / ".env")

    ap = argparse.ArgumentParser(description="Store important insider trades in TimescaleDB.")
    ap.add_argument("--min-value", type=float, default=float(os.getenv("INSIDER_MIN_VALUE", "50000")),
                    help="minimum transaction value in USD (default 50000)")
    args = ap.parse_args()

    key = os.getenv("FMP_API_KEY")
    if not key:
        print("Error: FMP_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1

    from a2a_finance import tiger_client
    if not tiger_client.enabled():
        print("Error: TimescaleDB not configured (TIGER_DATABASE_* + timescale extra).",
              file=sys.stderr)
        return 1

    feed = fetch_latest(key)
    rows = []
    for rec in feed:
        ok, value = _important(rec, args.min_value)
        if not ok:
            continue
        ts = _to_ts(rec.get("transactionDate"))
        if ts is None:
            continue
        rows.append({
            "trade_id": _trade_id(rec),
            "transaction_date": ts,
            "symbol": rec.get("symbol"),
            "company_cik": rec.get("companyCik"),
            "reporting_cik": rec.get("reportingCik"),
            "reporting_name": rec.get("reportingName"),
            "type_of_owner": rec.get("typeOfOwner"),
            "transaction_type": rec.get("transactionType"),
            "acquisition_disposition": rec.get("acquisitionOrDisposition"),
            "securities_transacted": rec.get("securitiesTransacted"),
            "price": rec.get("price"),
            "value": round(value, 2),
            "securities_owned": rec.get("securitiesOwned"),
            "form_type": rec.get("formType"),
            "security_name": rec.get("securityName"),
            "filing_date": rec.get("filingDate"),
            "url": rec.get("url"),
        })

    written = tiger_client.save_insider_trades(rows)
    print(f"▸ feed={len(feed)}  important={len(rows)}  upserted={written} "
          f"(min_value=${args.min_value:,.0f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
