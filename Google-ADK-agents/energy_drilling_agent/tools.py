"""EIA Drilling Productivity Report (DPR) tools via the OilPrice API.

Covers DUC (Drilled but Uncompleted) well inventories and per-rig oil/gas
productivity by basin, from the EIA Drilling Productivity Report (released ~15th
of each month).

**These are PREMIUM endpoints** — OilPrice's "Scale" plan
(``/v1/ei/drilling_productivities/*``). They are implemented in full so anyone
reproducing this work with a Scale key can run them unchanged. On a free/lower
tier the API answers 401/402/403 and every tool returns a clear
``{"error": "… PREMIUM endpoint …"}`` rather than data — the agent reports the
gap instead of crashing.
"""

from __future__ import annotations

import os

import httpx

BASE = "https://api.oilpriceapi.com/v1/ei/drilling_productivities"
_TIMEOUT = 20.0

# EIA DPR basins (primary product / region).
VALID_BASINS = [
    "permian",      # Oil — TX/NM
    "bakken",       # Oil — ND/MT
    "eagle_ford",   # Oil — TX
    "niobrara",     # Oil — CO/WY
    "appalachia",   # Gas — PA/WV/OH
    "anadarko",     # Oil/Gas — OK
    "haynesville",  # Gas — LA/TX
]


def _get(path: str = "", params: dict | None = None) -> dict:
    """GET a drilling-productivity endpoint, returning parsed JSON or an error dict.

    Premium-gated: a 401/402/403 becomes a clear "premium endpoint" error so the
    agent can report that the Scale plan is required rather than crash.
    """
    key = os.environ.get("OILPRICE_API_KEY")
    if not key:
        return {"error": "OILPRICE_API_KEY not set — add it to finance_coordinator/.env"}
    url = f"{BASE}/{path}".rstrip("/")
    try:
        resp = httpx.get(url, params=params or {},
                         headers={"Authorization": f"Token {key}"}, timeout=_TIMEOUT)
        if resp.status_code in (401, 402, 403):
            return {
                "error": "EIA Drilling Productivity is a PREMIUM OilPrice endpoint "
                         "(requires the 'Scale' plan).",
                "status": resp.status_code,
                "detail": resp.text[:200],
            }
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"OilPrice HTTP {e.response.status_code}", "detail": e.response.text[:200]}
    except Exception as e:
        return {"error": f"OilPrice request failed: {e}"}


def _data(resp: dict) -> dict:
    """Unwrap the ``{ "data": … }`` envelope, or pass an error dict through."""
    if isinstance(resp, dict) and "error" in resp:
        return resp
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp if isinstance(resp, dict) else {"error": "unexpected response shape"}


def list_drilling_reports(page: int = 1, per_page: int = 10) -> dict:
    """List paginated EIA Drilling Productivity reports (newest first).

    Args:
        page: Page number (1-based).
        per_page: Results per page (max 50).
    """
    return _data(_get("", {"page": max(1, page), "per_page": min(max(1, per_page), 50)}))


def get_latest_drilling_report() -> dict:
    """Latest DPR: total DUC wells + per-basin rig productivity (oil bpd/rig, gas mcf/rig)."""
    return _data(_get("latest"))


def get_drilling_summary() -> dict:
    """Current metrics summary: total DUC, average oil/gas productivity, per-basin headline."""
    return _data(_get("summary"))


def get_duc_wells() -> dict:
    """DUC (Drilled but Uncompleted) well counts by basin, with month-over-month change."""
    return _data(_get("duc_wells"))


def get_drilling_by_basin(basins: str = "", months: int = 12) -> dict:
    """DUC + productivity series filtered by basin.

    Args:
        basins: Comma-separated basin names (see VALID_BASINS), e.g.
            "permian,bakken". Empty = all basins.
        months: Months of data (max 60).
    """
    params: dict = {"months": min(max(1, months), 60)}
    if basins:
        params["basins"] = basins
    return _data(_get("by_basin", params))


def get_drilling_historical(basin: str, months: int = 24) -> dict:
    """Historical DUC / productivity / legacy-decline series for one basin.

    Args:
        basin: A single basin name (see VALID_BASINS), e.g. "permian".
        months: Months of history (max 120).
    """
    if basin and basin.lower() not in VALID_BASINS:
        return {"error": f"unknown basin '{basin}'; valid: {VALID_BASINS}"}
    return _data(_get("historical", {"basin": basin.lower(), "months": min(max(1, months), 120)}))


def get_drilling_trends(months: int = 12) -> dict:
    """Cross-basin trend analysis: DUC change/direction + productivity trend per basin.

    Args:
        months: Analysis period in months (min 3, max 36).
    """
    return _data(_get("trends", {"months": min(max(3, months), 36)}))


def get_drilling_report(report_id: str) -> dict:
    """Fetch one specific DPR by its id.

    Args:
        report_id: The report id from list_drilling_reports.
    """
    return _data(_get(str(report_id)))
