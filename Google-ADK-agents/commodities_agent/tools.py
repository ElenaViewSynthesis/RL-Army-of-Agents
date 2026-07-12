"""OilPrice API tools for the commodities agent.

Wraps https://api.oilpriceapi.com/v1 — a catalog of 460+ commodity codes plus
latest and historical prices. Key is read from ``OILPRICE_API_KEY``.

Catalog/search results are bounded (the full catalog is 460+ rows, which would
swamp the model's context), and every function returns a JSON-serializable dict
with an ``{"error": ...}`` fallback so the agent reports gaps instead of crashing.
"""

from __future__ import annotations

import os
import time

import httpx

BASE = "https://api.oilpriceapi.com/v1"
_TIMEOUT = 20.0

# Curated "Fuse Energy" watchlist — codes most useful to a UK/London energy
# retailer, grouped by theme. See commodities_agent/FUSE_ENERGY_WATCHLIST.md.
FUSE_WATCHLIST: list[tuple[str, str, str]] = [
    ("NATURAL_GAS_GBP", "UK Natural Gas", "gas"),
    ("NATURAL_GAS_TTF_SPOT_EUR", "TTF Natural Gas Spot", "gas"),
    ("BRENT_CRUDE_USD", "Brent Crude", "petroleum"),
    ("GASOIL_USD", "ICE Gasoil (Rotterdam)", "petroleum"),
    ("UK_CARBON_GBP", "UK Carbon (UK ETS)", "carbon"),
    ("EU_CARBON_EUR", "EU Carbon (EU ETS)", "carbon"),
    ("NEWCASTLE_COAL_USD", "Newcastle Coal (API6)", "coal"),
]
_MAX_ROWS = 30  # cap catalog/search output so it fits the model context


def _get(path: str, params: dict | None = None) -> dict | list:
    key = os.environ.get("OILPRICE_API_KEY")
    if not key:
        return {"error": "OILPRICE_API_KEY not set — add it to finance_coordinator/.env"}
    try:
        resp = httpx.get(
            f"{BASE}/{path}",
            params=params or {},
            headers={"Authorization": f"Token {key}"},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 401:
            return {"error": "OilPrice API auth failed (401) — check OILPRICE_API_KEY"}
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"OilPrice HTTP {e.response.status_code}", "detail": e.response.text[:200]}
    except Exception as e:
        return {"error": f"OilPrice request failed: {e}"}


def _catalog() -> list | dict:
    data = _get("commodities")
    if isinstance(data, dict) and "error" in data:
        return data
    return (data.get("data") or {}).get("commodities", []) if isinstance(data, dict) else []


def list_commodities(category: str = "") -> dict:
    """List available commodity codes, optionally filtered by category.

    Categories include: oil, gas, coal, metal, forex, refined_products,
    petrochemical, marine_fuel, emissions, macro_indicators, and more.

    Args:
        category: Optional category to filter by (case-insensitive). Empty = all
            categories (results are capped).
    """
    items = _catalog()
    if isinstance(items, dict):
        return items  # error
    if category:
        items = [c for c in items if c.get("category", "").lower() == category.lower()]
    total = len(items)
    rows = [
        {"code": c["code"], "name": c["name"], "category": c["category"], "unit": c.get("unit")}
        for c in items[:_MAX_ROWS]
    ]
    return {"count": total, "showing": len(rows), "truncated": total > len(rows), "commodities": rows}


def search_commodities(query: str) -> dict:
    """Search the commodity catalog by name or code (case-insensitive substring).

    Args:
        query: Text to match against commodity name or code, e.g. "brent", "gas".
    """
    items = _catalog()
    if isinstance(items, dict):
        return items  # error
    q = query.lower()
    matches = [
        {"code": c["code"], "name": c["name"], "category": c["category"], "unit": c.get("unit")}
        for c in items
        if q in c["name"].lower() or q in c["code"].lower()
    ]
    total = len(matches)
    return {"query": query, "count": total, "truncated": total > _MAX_ROWS, "matches": matches[:_MAX_ROWS]}


def get_commodity_price(code: str) -> dict:
    """Get the latest price for a commodity by its code.

    Args:
        code: Commodity code, e.g. "BRENT_CRUDE_USD", "WTI_USD", "NATURAL_GAS_USD".
    """
    data = _get("prices/latest", {"by_code": code.upper()})
    if isinstance(data, dict) and "error" in data:
        return data
    d = data.get("data", {}) if isinstance(data, dict) else {}
    return {
        "code": d.get("code", code.upper()),
        "price": d.get("price"),
        "formatted": d.get("formatted"),
        "currency": d.get("currency"),
        "unit": d.get("unit"),
        "type": d.get("type"),
        "updated_at": d.get("updated_at") or d.get("created_at"),
    }


def get_commodity_history(code: str, period: str = "past_week") -> dict:
    """Get historical prices for a commodity over a period.

    Args:
        code: Commodity code, e.g. "BRENT_CRUDE_USD".
        period: One of "past_day", "past_week", "past_month", "past_year".
    """
    valid = {"past_day", "past_week", "past_month", "past_year"}
    if period not in valid:
        return {"error": f"period must be one of {sorted(valid)}"}
    data = _get(f"prices/{period}", {"by_code": code.upper()})
    if isinstance(data, dict) and "error" in data:
        return data
    prices = (data.get("data") or {}).get("prices", []) if isinstance(data, dict) else []
    slim = [{"price": p.get("price"), "at": p.get("created_at")} for p in prices[:60]]
    return {"code": code.upper(), "period": period, "count": len(prices), "prices": slim}


def list_fuse_watchlist() -> dict:
    """Return live prices for the curated Fuse Energy watchlist in one call.

    Fetches the latest price for each code most useful to a UK/London energy
    retailer — UK & TTF gas, Brent, ICE Gasoil, UK & EU carbon, and Newcastle
    coal — grouped by theme (gas, petroleum, carbon, coal). Takes no arguments.
    """
    # One request per code; throttle to respect the 60/min (1/sec) rate limit.
    themes: dict[str, list] = {}
    for i, (code, label, theme) in enumerate(FUSE_WATCHLIST):
        if i:
            time.sleep(1.05)
        p = get_commodity_price(code)
        themes.setdefault(theme, []).append({
            "code": code,
            "label": label,
            "price": p.get("price"),
            "formatted": p.get("formatted"),
            "currency": p.get("currency"),
            "unit": p.get("unit"),
            "error": p.get("error"),
        })
    return {"watchlist": "fuse_energy", "count": len(FUSE_WATCHLIST), "by_theme": themes}
