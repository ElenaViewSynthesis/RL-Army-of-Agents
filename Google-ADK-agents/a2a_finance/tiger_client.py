"""Optional TimescaleDB (Tiger Cloud) persistence for commodity price time-series.

The numeric time-series half of the system lives here: a TimescaleDB
**hypertable** ``commodity_prices`` (partitioned on ``ts``), the natural home for
dense price data — while Supabase keeps the narrative/embedding side
(``agent_responses``). "Numbers -> TimescaleDB, words -> Supabase."

**Write path: direct Postgres via psycopg.** Config comes from
``finance_coordinator/.env``: either ``TIGER_DATABASE_URL`` (if already expanded)
or the individual ``TIGER_DATABASE_*`` parts (host/port/name/user/password).

**Graceful by design.** If no connection is configured, ``psycopg`` isn't
installed (the optional ``timescale`` extra), or the DB is unreachable, every
function is a silent no-op. Seeding/persistence must never break a run.

Schema (created out-of-band; see scripts)::

    commodity_prices(ts timestamptz, code text, source text, source_symbol text,
                     name text, price float8, currency text, unit text,
                     change float8, PRIMARY KEY (code, source, ts))
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("a2a_finance.tiger_client")

# Self-contained .env autoloader (same pattern as storage.py) so a bare import
# works without an explicit load_dotenv. Only fills vars that are absent.
_ENV_PATH = Path(__file__).resolve().parent.parent / "finance_coordinator" / ".env"
_env_loaded = False


def _ensure_env_loaded() -> None:
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if os.getenv("TIGER_DATABASE_URL") or os.getenv("TIGER_DATABASE_HOST"):
        return
    try:
        for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_conn = None
_conn_lock = threading.Lock()
_disabled = False


def _db_url() -> Optional[str]:
    """Resolve the connection string from TIGER_DATABASE_URL or the parts.

    Prefer TIGER_DATABASE_URL, but only if it's already expanded (dotenv may not
    interpolate ``${...}``); otherwise build it from the individual components.
    """
    _ensure_env_loaded()
    url = os.getenv("TIGER_DATABASE_URL")
    if url and "${" not in url:
        return url.strip()
    host = os.getenv("TIGER_DATABASE_HOST")
    user = os.getenv("TIGER_DATABASE_USER")
    pwd = os.getenv("TIGER_DATABASE_PASSWORD")
    name = os.getenv("TIGER_DATABASE_NAME", "tsdb")
    port = os.getenv("TIGER_DATABASE_PORT", "5432")
    if host and user and pwd:
        return f"postgresql://{user}:{pwd}@{host}:{port}/{name}?sslmode=require"
    return None


def enabled() -> bool:
    """True if a TimescaleDB connection is configured and psycopg importable."""
    if _disabled or not _db_url():
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def _get_conn():
    global _conn, _disabled
    if _disabled:
        return None
    url = _db_url()
    if not url:
        _disabled = True
        return None
    with _conn_lock:
        if _conn is not None and not _conn.closed:
            return _conn
        try:
            import psycopg
        except ImportError:
            logger.warning("TIGER_DATABASE_* set but psycopg not installed; "
                           "install the 'timescale' extra. Persistence disabled.")
            _disabled = True
            return None
        try:
            _conn = psycopg.connect(url, autocommit=True, connect_timeout=15)
            logger.info("TimescaleDB connected.")
            return _conn
        except Exception as e:
            logger.warning("TimescaleDB connect failed (disabled): %s", e)
            _disabled = True
            return None


_UPSERT = """
INSERT INTO commodity_prices
    (ts, code, source, source_symbol, name, price, currency, unit, change)
VALUES
    (%(ts)s, %(code)s, %(source)s, %(source_symbol)s, %(name)s,
     %(price)s, %(currency)s, %(unit)s, %(change)s)
ON CONFLICT (code, source, ts) DO UPDATE SET
    price = EXCLUDED.price, currency = EXCLUDED.currency, unit = EXCLUDED.unit,
    change = EXCLUDED.change, source_symbol = EXCLUDED.source_symbol,
    name = EXCLUDED.name
"""

_COLS = ("ts", "code", "source", "source_symbol", "name",
         "price", "currency", "unit", "change")


def save_prices(rows: Iterable[dict]) -> int:
    """Upsert commodity price rows into the hypertable. Returns rows written.

    Idempotent on ``(code, source, ts)`` — safe to re-run overlapping windows.
    Each row needs at least ``ts``, ``code``, ``source``; the rest default to
    None. No-op (returns 0) if TimescaleDB isn't configured/reachable.
    """
    if not enabled():
        return 0
    payload = []
    for r in rows:
        if not r.get("ts") or not r.get("code") or not r.get("source"):
            continue
        payload.append({k: r.get(k) for k in _COLS})
    if not payload:
        return 0
    return _executemany(_UPSERT, payload)


_INSIDER_COLS = (
    "trade_id", "transaction_date", "symbol", "company_cik", "reporting_cik",
    "reporting_name", "type_of_owner", "transaction_type", "acquisition_disposition",
    "securities_transacted", "price", "value", "securities_owned", "form_type",
    "security_name", "filing_date", "url",
)

_INSIDER_UPSERT = """
INSERT INTO insider_trades
    (trade_id, transaction_date, symbol, company_cik, reporting_cik, reporting_name,
     type_of_owner, transaction_type, acquisition_disposition, securities_transacted,
     price, value, securities_owned, form_type, security_name, filing_date, url)
VALUES
    (%(trade_id)s, %(transaction_date)s, %(symbol)s, %(company_cik)s, %(reporting_cik)s,
     %(reporting_name)s, %(type_of_owner)s, %(transaction_type)s, %(acquisition_disposition)s,
     %(securities_transacted)s, %(price)s, %(value)s, %(securities_owned)s, %(form_type)s,
     %(security_name)s, %(filing_date)s, %(url)s)
ON CONFLICT (transaction_date, trade_id) DO UPDATE SET
    securities_owned = EXCLUDED.securities_owned, url = EXCLUDED.url
"""


def save_insider_trades(rows: Iterable[dict]) -> int:
    """Upsert insider-trade rows into the hypertable. Returns rows written.

    Idempotent on ``(transaction_date, trade_id)`` — re-polling the same feed
    upserts rather than duplicating. Each row needs at least ``trade_id``,
    ``transaction_date``, ``symbol``. No-op if TimescaleDB isn't configured.
    """
    if not enabled():
        return 0
    payload = []
    for r in rows:
        if not r.get("trade_id") or not r.get("transaction_date") or not r.get("symbol"):
            continue
        payload.append({k: r.get(k) for k in _INSIDER_COLS})
    if not payload:
        return 0
    return _executemany(_INSIDER_UPSERT, payload)


_MARINE_COLS = ("snapshot_ts", "code", "name", "country", "region", "major_port",
                "latitude", "longitude", "fuel_services", "trading_hours")

_MARINE_UPSERT = """
INSERT INTO marine_ports
    (snapshot_ts, code, name, country, region, major_port,
     latitude, longitude, fuel_services, trading_hours)
VALUES
    (%(snapshot_ts)s, %(code)s, %(name)s, %(country)s, %(region)s, %(major_port)s,
     %(latitude)s, %(longitude)s, %(fuel_services)s, %(trading_hours)s)
ON CONFLICT (code, snapshot_ts) DO UPDATE SET
    name = EXCLUDED.name, country = EXCLUDED.country, region = EXCLUDED.region,
    major_port = EXCLUDED.major_port, latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude, fuel_services = EXCLUDED.fuel_services,
    trading_hours = EXCLUDED.trading_hours
"""


def save_marine_ports(rows: Iterable[dict]) -> int:
    """Upsert marine-fuel-port snapshot rows into the columnstore hypertable.

    Each row needs at least ``snapshot_ts``, ``code``. ``fuel_services`` is a
    Python list (mapped to a Postgres ``text[]``). No-op if TimescaleDB isn't
    configured. Idempotent on ``(code, snapshot_ts)``.
    """
    if not enabled():
        return 0
    payload = []
    for r in rows:
        if not r.get("snapshot_ts") or not r.get("code"):
            continue
        payload.append({k: r.get(k) for k in _MARINE_COLS})
    if not payload:
        return 0
    return _executemany(_MARINE_UPSERT, payload)


def _executemany(sql: str, payload: list) -> int:
    """Run a batch upsert; return rows written (0 on failure). Never raises."""
    conn = _get_conn()
    if conn is None:
        return 0
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, payload)
        return len(payload)
    except Exception as e:
        global _conn
        with _conn_lock:
            try:
                if _conn is not None:
                    _conn.close()
            except Exception:
                pass
            _conn = None
        logger.warning("TimescaleDB write failed (skipped): %s", e)
        return 0
