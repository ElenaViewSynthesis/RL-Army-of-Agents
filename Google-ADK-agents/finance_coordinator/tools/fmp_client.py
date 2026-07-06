"""Shared Financial Modeling Prep (FMP) client for the finance tools.

Mirrors the ``/stable`` calling convention used by the sibling
``Equity-Research-agent`` (``?symbol=...&apikey=...``). The API key is read
from ``FMP_API_KEY`` in the environment — ADK loads the ``.env`` next to the
agent package, so put it in ``finance_coordinator/.env``.

Tools call :func:`fmp_get` and get back either the parsed JSON or a dict with
an ``"error"`` key. Tools return errors rather than raising so the model can
reason about missing data instead of the whole turn failing.
"""

from __future__ import annotations

import os

import httpx

FMP_STABLE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 20.0


def fmp_get(path: str, params: dict | None = None) -> list | dict:
    """GET an FMP `/stable` endpoint and return parsed JSON, or an error dict.

    Args:
        path: Endpoint path under `/stable`, e.g. "profile" or "stock-peers".
        params: Query params (excluding apikey), typically ``{"symbol": ...}``.

    Returns:
        The parsed JSON (usually a list of records), or ``{"error": "..."}``
        on a missing key, premium-tier gate (HTTP 402), or request failure.
    """
    key = os.environ.get("FMP_API_KEY")
    if not key:
        return {"error": "FMP_API_KEY not set — add it to finance_coordinator/.env"}

    url = f"{FMP_STABLE}/{path}"
    try:
        resp = httpx.get(url, params={**(params or {}), "apikey": key}, timeout=_TIMEOUT)
        if resp.status_code == 402:
            return {"error": "premium FMP endpoint — not available on the free tier", "path": path}
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"FMP HTTP {e.response.status_code}", "detail": e.response.text[:200]}
    except Exception as e:  # network, timeout, JSON decode
        return {"error": f"FMP request failed: {e}"}


def first_record(data: list | dict) -> dict:
    """Normalize an FMP list response to its first record.

    FMP symbol-keyed endpoints return a single-element list. Pass through an
    error dict unchanged, and report an empty response explicitly.
    """
    if isinstance(data, dict):
        return data  # error dict
    if isinstance(data, list):
        return data[0] if data else {"error": "no data returned by FMP"}
    return {"error": "unexpected FMP response shape"}
