"""ICE Gas Oil futures — a separate, optional OilPrice tool group (PREMIUM).

Wraps `GET /v1/futures/ice-gasoil/*` (current, historical, OHLC, intraday,
spreads, curve, spread-history). Requires OilPrice's **"Futures Data"**
entitlement; without it the API answers `403` and every tool returns a
**structured** `futures_data_not_entitled` error. So the group can be wired in
now and light up when the entitlement is added — no agent restructuring.

Opt in by adding `FUTURES_TOOLS` to an agent (the commodities agent does this
when `OILPRICE_FUTURES=1`). Standalone / testable: only depends on `httpx`.

**Not stored in `commodity_prices`.** That table is spot/index prices. Futures
are contract-level (contract/delivery month, expiry, curve timestamp, OHLC, open
interest) and need their own schema — see `GASOIL_FUTURES_DDL` below (a dedicated
`gasoil_futures` hypertable; create + wire persistence when the entitlement is
active).
"""

from __future__ import annotations

import os

import httpx

BASE = "https://api.oilpriceapi.com/v1/futures/ice-gasoil"
_TIMEOUT = 20.0

# Dedicated schema for contract-level futures data (do NOT reuse commodity_prices).
# Create when the Futures Data entitlement is active and persistence is wanted.
GASOIL_FUTURES_DDL = """
CREATE TABLE IF NOT EXISTS gasoil_futures (
    curve_ts        timestamptz NOT NULL,   -- curve/snapshot timestamp (partition)
    contract_month  text        NOT NULL,   -- delivery month, e.g. 2026-01
    last_price      double precision,
    open double precision, high double precision, low double precision, close double precision,
    volume bigint, open_interest bigint,
    currency text, unit text,
    change_percent  double precision,
    days_to_expiry  integer,
    expiry_date     date,
    is_front_month  boolean,
    contract_status text,
    PRIMARY KEY (contract_month, curve_ts)
);
SELECT create_hypertable('gasoil_futures', 'curve_ts', if_not_exists => TRUE);
"""


def _get(path: str = "", params: dict | None = None) -> dict:
    """GET an ICE-gasoil futures endpoint, normalizing failures to structured errors.

    Distinguishes: missing key, 403 (not entitled), 401 (auth), 429 (rate limit),
    other 4xx/5xx (upstream error), and network/timeout (upstream unreachable).
    A successful call returns the parsed JSON unchanged.
    """
    key = os.environ.get("OILPRICE_API_KEY")
    if not key:
        return {"error": "missing_api_key",
                "message": "OILPRICE_API_KEY not set — add it to finance_coordinator/.env"}
    url = f"{BASE}/{path}".rstrip("/")
    try:
        resp = httpx.get(url, params=params or {},
                         headers={"Authorization": f"Token {key}"}, timeout=_TIMEOUT)
    except httpx.TimeoutException:
        return {"error": "upstream_timeout", "message": "OilPrice request timed out."}
    except httpx.RequestError as e:
        return {"error": "upstream_unreachable", "message": f"OilPrice request failed: {e}"}

    if resp.status_code == 403:
        return {"error": "futures_data_not_entitled",
                "message": "Futures Data is not included in the current OilPrice API plan."}
    if resp.status_code == 401:
        return {"error": "unauthorized",
                "message": "OilPrice authentication failed (401) — check OILPRICE_API_KEY."}
    if resp.status_code == 429:
        return {"error": "rate_limited",
                "message": "OilPrice rate limit hit (429) — retry later.",
                "retry_after": resp.headers.get("Retry-After")}
    if resp.status_code >= 400:
        return {"error": "upstream_error", "status": resp.status_code,
                "message": resp.text[:200]}
    try:
        return resp.json()
    except ValueError:
        return {"error": "bad_response", "message": "OilPrice returned a non-JSON body."}


# ── Tools (premium; structured error on the free tier) ───────────────────────
def get_gasoil_futures() -> dict:
    """Current ICE Gas Oil futures: per contract-month last/OHLC/volume/open-interest."""
    return _get("")


def get_gasoil_historical(contract_month: str = "") -> dict:
    """Historical ICE Gas Oil futures prices.

    Args:
        contract_month: Optional delivery month filter, e.g. "2026-01". Empty = all.
    """
    return _get("historical", {"contract_month": contract_month} if contract_month else None)


def get_gasoil_ohlc(contract_month: str = "") -> dict:
    """Daily OHLC for ICE Gas Oil futures.

    Args:
        contract_month: Optional delivery month filter, e.g. "2026-01".
    """
    return _get("ohlc", {"contract_month": contract_month} if contract_month else None)


def get_gasoil_intraday(contract_month: str = "") -> dict:
    """Intraday (5-minute) ICE Gas Oil futures prices.

    Args:
        contract_month: Optional delivery month filter, e.g. "2026-01".
    """
    return _get("intraday", {"contract_month": contract_month} if contract_month else None)


def get_gasoil_spreads() -> dict:
    """Calendar spreads between ICE Gas Oil futures contract months."""
    return _get("spreads")


def get_gasoil_curve() -> dict:
    """The ICE Gas Oil futures curve (term structure across contract months)."""
    return _get("curve")


def get_gasoil_spread_history(spread: str = "") -> dict:
    """History of an ICE Gas Oil calendar spread.

    Args:
        spread: Optional spread identifier (e.g. "2026-01/2026-02").
    """
    return _get("spread-history", {"spread": spread} if spread else None)


# The optional tool group — extend an agent's `tools=[…]` with this.
FUTURES_TOOLS = [
    get_gasoil_futures,
    get_gasoil_historical,
    get_gasoil_ohlc,
    get_gasoil_intraday,
    get_gasoil_spreads,
    get_gasoil_curve,
    get_gasoil_spread_history,
]
