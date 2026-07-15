#!/usr/bin/env python3
"""Seed the TimescaleDB `commodity_prices` hypertable with daily energy prices.

Designed for a **daily cron within the OilPrice 200-requests/day budget**: each
run pulls the last few days of daily history (default 4-day window) per code and
**upserts** — overlapping windows are idempotent (PK is code, source, ts), so a
missed day self-heals on the next run and re-runs never duplicate.

Sources
- OilPrice API  -> historical daily points (the multi-day series). ~1 req/code.
- FMP (/stable) -> a current snapshot for the overlapping symbols (WTI/Brent/gas/
  heating oil/gasoline). FMP *historical* commodities are premium, so FMP only
  contributes today's price as a cross-source. ~1 req/symbol.

Self-contained (httpx + psycopg via tiger_client) — no ADK import, so it runs
anywhere. Reads keys from finance_coordinator/.env.

    uv run --extra timescale python seed_timescale_prices.py            # all, 4-day
    uv run --extra timescale python seed_timescale_prices.py --days 3 --no-fmp
    uv run --extra timescale python seed_timescale_prices.py WTI BRENT   # subset
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

OILPRICE_BASE = "https://api.oilpriceapi.com/v1"
FMP_STABLE = "https://financialmodelingprep.com/stable"
_THROTTLE = 1.1  # seconds between OilPrice calls (respect ~60/min)

# normalized code -> (display name, OilPrice code, FMP symbol|None, unit)
UNIVERSE: dict[str, tuple[str, str, str | None, str]] = {
    "WTI":         ("WTI Crude",              "WTI_USD",                  "CLUSD", "barrel"),
    "BRENT":       ("Brent Crude",            "BRENT_CRUDE_USD",          "BZUSD", "barrel"),
    "NATGAS":      ("Henry Hub Natural Gas",  "NATURAL_GAS_USD",          "NGUSD", "MMBtu"),
    "HEATING_OIL": ("Heating Oil",            "HEATING_OIL_USD",          "HOUSD", "gallon"),
    "GASOLINE":    ("RBOB Gasoline",          "GASOLINE_USD",             "RBUSD", "gallon"),
    "GASOIL":      ("ICE Gasoil",             "GASOIL_USD",               None,    "tonne"),
    "NATGAS_UK":   ("UK NBP Natural Gas",     "NATURAL_GAS_GBP",          None,    "therm"),
    "NATGAS_TTF":  ("TTF Natural Gas Spot",   "NATURAL_GAS_TTF_SPOT_EUR", None,    "MWh"),
    "COAL_NEWC":   ("Newcastle Coal (API6)",  "NEWCASTLE_COAL_USD",       None,    "tonne"),
    "CARBON_UK":   ("UK Carbon (UK ETS)",     "UK_CARBON_GBP",            None,    "tonne"),
    "CARBON_EU":   ("EU Carbon (EU ETS)",     "EU_CARBON_EUR",            None,    "tonne"),
}


def _currency_of(oilprice_code: str) -> str | None:
    tail = oilprice_code.rsplit("_", 1)[-1]
    return tail if tail in ("USD", "GBP", "EUR") else None


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# ── OilPrice (historical) ─────────────────────────────────────────────────────
def oilprice_history(code: str, key: str, period: str = "past_week") -> list[dict]:
    try:
        r = httpx.get(f"{OILPRICE_BASE}/prices/{period}", params={"by_code": code},
                      headers={"Authorization": f"Token {key}"}, timeout=20)
        if r.status_code != 200:
            print(f"    oilprice {code}: HTTP {r.status_code}", file=sys.stderr)
            return []
        return (r.json().get("data") or {}).get("prices", []) or []
    except Exception as e:
        print(f"    oilprice {code}: {e}", file=sys.stderr)
        return []


# ── FMP (current snapshot) ────────────────────────────────────────────────────
def fmp_quote(symbol: str, key: str) -> dict | None:
    try:
        r = httpx.get(f"{FMP_STABLE}/quote", params={"symbol": symbol, "apikey": key}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except Exception:
        return None


def seed(codes: list[str], days: int, use_fmp: bool, budget: int) -> int:
    op_key = os.getenv("OILPRICE_API_KEY")
    fmp_key = os.getenv("FMP_API_KEY")
    if not op_key:
        print("Error: OILPRICE_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 0

    from a2a_finance import tiger_client
    if not tiger_client.enabled():
        print("Error: TimescaleDB not configured/reachable (TIGER_DATABASE_* + "
              "the 'timescale' extra).", file=sys.stderr)
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: list[dict] = []
    op_requests = 0

    for i, norm in enumerate(codes):
        name, op_code, fmp_sym, unit = UNIVERSE[norm]
        currency = _currency_of(op_code)

        # OilPrice historical window
        if op_requests >= budget:
            print(f"  [budget] stopping OilPrice fetches at {op_requests} requests", file=sys.stderr)
        else:
            if i:
                time.sleep(_THROTTLE)
            points = oilprice_history(op_code, op_key)
            op_requests += 1
            kept = 0
            for p in points:
                ts = _parse_ts(p.get("created_at") or p.get("at"))
                price = p.get("price")
                if ts is None or price is None or ts < cutoff:
                    continue
                rows.append({"ts": ts, "code": norm, "source": "oilprice",
                             "source_symbol": op_code, "name": name, "price": price,
                             "currency": currency, "unit": unit, "change": None})
                kept += 1
            print(f"  {norm:11s} oilprice: {kept} point(s) in window", file=sys.stderr)

        # FMP current snapshot (cross-source), if it maps and is enabled
        if use_fmp and fmp_sym and fmp_key:
            q = fmp_quote(fmp_sym, fmp_key)
            if q and q.get("price") is not None:
                rows.append({"ts": datetime.now(timezone.utc), "code": norm, "source": "fmp",
                             "source_symbol": fmp_sym, "name": name, "price": q.get("price"),
                             "currency": "USD", "unit": unit, "change": q.get("change")})
                print(f"  {norm:11s} fmp:      {q.get('price')}", file=sys.stderr)

    written = tiger_client.save_prices(rows)
    print(f"\n▸ upserted {written} row(s) into commodity_prices "
          f"({op_requests} OilPrice request(s) used of {budget} budget).", file=sys.stderr)
    return written


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / "finance_coordinator" / ".env")

    ap = argparse.ArgumentParser(description="Seed TimescaleDB with daily commodity prices.")
    ap.add_argument("codes", nargs="*", help=f"subset of {list(UNIVERSE)}; empty = all")
    ap.add_argument("--days", type=int, default=4, help="history window in days (default 4)")
    ap.add_argument("--no-fmp", action="store_true", help="skip the FMP current snapshot")
    ap.add_argument("--budget", type=int, default=60, help="max OilPrice requests this run")
    args = ap.parse_args()

    codes = [c.upper() for c in args.codes] or list(UNIVERSE)
    unknown = [c for c in codes if c not in UNIVERSE]
    if unknown:
        print(f"Unknown code(s) {unknown}; valid: {list(UNIVERSE)}", file=sys.stderr)
        return 1

    seed(codes, days=args.days, use_fmp=not args.no_fmp, budget=args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
