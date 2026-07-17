"""Mocked tests for the ICE Gas Oil futures tool group (oilprice_futures).

No network: httpx.get is monkeypatched. Covers a successful response and the
structured error normalization (403 not-entitled, 401, 429, timeout, missing key).
"""

from __future__ import annotations

import httpx
import pytest

import oilprice_futures as f


class FakeResponse:
    def __init__(self, status_code, json_body=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    # Every test except the missing-key one needs a key present.
    monkeypatch.setenv("OILPRICE_API_KEY", "test-key")


def _patch_get(monkeypatch, response=None, exc=None):
    def fake_get(url, params=None, headers=None, timeout=None):
        if exc is not None:
            raise exc
        return response
    monkeypatch.setattr(f.httpx, "get", fake_get)


# ── success ──────────────────────────────────────────────────────────────────
SAMPLE = {
    "commodity": "GASOIL_FUTURES",
    "contracts": [
        {"contract_month": "2026-01", "last_price": 685.50, "currency": "USD",
         "unit": "metric_ton", "is_front_month": False, "days_to_expiry": 8},
    ],
    "metadata": {"total_contracts": 12, "front_month_contract": "2026-02"},
}


def test_current_success(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(200, SAMPLE))
    out = f.get_gasoil_futures()
    assert out["commodity"] == "GASOIL_FUTURES"
    assert out["contracts"][0]["contract_month"] == "2026-01"
    assert "error" not in out


def test_curve_success(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(200, {"data": {"curve": []}}))
    assert f.get_gasoil_curve() == {"data": {"curve": []}}


# ── structured errors ────────────────────────────────────────────────────────
def test_403_not_entitled(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(403, text='{"error":"Feature access required"}'))
    out = f.get_gasoil_futures()
    assert out == {
        "error": "futures_data_not_entitled",
        "message": "Futures Data is not included in the current OilPrice API plan.",
    }


def test_401_unauthorized(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(401))
    assert f.get_gasoil_ohlc()["error"] == "unauthorized"


def test_429_rate_limited(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(429, headers={"Retry-After": "30"}))
    out = f.get_gasoil_spreads()
    assert out["error"] == "rate_limited"
    assert out["retry_after"] == "30"


def test_500_upstream_error(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(500, text="boom"))
    out = f.get_gasoil_curve()
    assert out["error"] == "upstream_error"
    assert out["status"] == 500


def test_timeout(monkeypatch):
    _patch_get(monkeypatch, exc=httpx.TimeoutException("slow"))
    assert f.get_gasoil_historical()["error"] == "upstream_timeout"


def test_connection_error(monkeypatch):
    _patch_get(monkeypatch, exc=httpx.ConnectError("no route"))
    assert f.get_gasoil_intraday()["error"] == "upstream_unreachable"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OILPRICE_API_KEY", raising=False)
    assert f.get_gasoil_futures()["error"] == "missing_api_key"


def test_tool_group_exposed():
    # The optional group is a list of 7 callables (wired into an agent on opt-in).
    assert len(f.FUTURES_TOOLS) == 7
    assert all(callable(t) for t in f.FUTURES_TOOLS)
