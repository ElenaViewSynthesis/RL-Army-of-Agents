"""Optional Postgres persistence for the A2A finance system.

Writes to the dedicated Supabase/Postgres project (``a2a-agents``) documented in
``resumefromdb.md``. Three tables, one envelope::

    agent_runs ──┬──< prices            (numbers: raw tool results)
                 └──< agent_responses   (words: LLM narrative / verdicts)

**Server-side, direct Postgres.** RLS is enabled with no policies, so anon keys
are blocked by design; a direct connection (the ``postgres`` role) bypasses RLS.
Point ``A2A_DB_URL`` at the connection string::

    A2A_DB_URL=postgresql://postgres:<PASSWORD>@db.ketwfywvgzpvhawzcrvi.supabase.co:5432/postgres

**Graceful by design.** If ``A2A_DB_URL`` is unset, ``psycopg`` isn't installed,
or the database is unreachable, every function is a silent no-op — the agents run
exactly as before, just without persistence. Persistence is a side effect; it
must never break a run.

Run-scoping uses a :class:`~contextvars.ContextVar`: ``start_run`` opens the
envelope and records its id on the context, and ``save_price`` / ``save_response``
attach to the current run without the caller threading an id through every tool
signature. Note the context var is per-process — in the multi-process A2A routing
path a run opened in the coordinator process is not visible inside a separate
specialist service process (each process opens its own run if it calls
``start_run``).
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("a2a_finance.storage")

# Current run id for this execution context; save_* default to it.
_run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "a2a_run_id", default=None
)

# One lazily-opened connection per process, guarded by a lock. Low write volume,
# so a single autocommit connection is plenty; we reconnect if it drops.
_conn = None
_conn_lock = threading.Lock()
_disabled = False  # set once if psycopg is missing or no URL is configured


def _db_url() -> Optional[str]:
    url = os.getenv("A2A_DB_URL")
    return url.strip() if url else None


def enabled() -> bool:
    """True if persistence is configured (``A2A_DB_URL`` set and psycopg importable)."""
    if _disabled or not _db_url():
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def _get_conn():
    """Return a live autocommit connection, or None if unavailable.

    Opens lazily and caches per process. Any failure disables persistence for the
    process (``_disabled``) so we don't retry-storm on a misconfigured URL.
    """
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
            logger.warning("A2A_DB_URL is set but psycopg is not installed; "
                           "persistence disabled. Add psycopg[binary].")
            _disabled = True
            return None
        try:
            _conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
            logger.info("A2A persistence connected.")
            return _conn
        except Exception as e:  # bad URL / password / network
            logger.warning("A2A persistence connect failed (disabled): %s", e)
            _disabled = True
            return None


def _execute(sql: str, params: tuple, *, fetch_one: bool = False):
    """Run a write; return the fetched row (if requested) or None. Never raises.

    On a dropped connection we reset the cached handle and retry once, so a stale
    socket doesn't permanently wedge persistence for the process.
    """
    if not _db_url():
        return None
    for attempt in (1, 2):
        conn = _get_conn()
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone() if fetch_one else None
        except Exception as e:
            global _conn
            # Drop the handle and retry once on the first failure (likely a stale
            # connection); give up quietly on the second.
            with _conn_lock:
                try:
                    if _conn is not None:
                        _conn.close()
                except Exception:
                    pass
                _conn = None
            if attempt == 2:
                logger.warning("A2A persistence write failed (skipped): %s", e)
                return None
    return None


def start_run(agent: str, subject: Optional[str] = None,
              prompt: Optional[str] = None) -> Optional[str]:
    """Open a run envelope and set it as the current run for this context.

    Args:
        agent: The agent/driver opening the run, e.g. ``"coordinator"``.
        subject: Ticker or commodity the run is about, e.g. ``"NVDA"``.
        prompt: The originating user prompt.

    Returns:
        The new run's uuid (str), or None when persistence is off.
    """
    if not enabled():
        return None
    row = _execute(
        "INSERT INTO agent_runs (agent, subject, prompt) VALUES (%s, %s, %s) "
        "RETURNING id",
        (agent, subject, prompt),
        fetch_one=True,
    )
    run_id = str(row[0]) if row else None
    if run_id:
        _run_id_var.set(run_id)
    return run_id


def current_run_id() -> Optional[str]:
    """The run id bound to the current context, if any."""
    return _run_id_var.get()


def save_price(code: str, price: Optional[float], currency: Optional[str] = None,
               unit: Optional[str] = None, source: Optional[str] = None,
               run_id: Optional[str] = None) -> None:
    """Record one price point against a run (no-op if persistence is off).

    Args:
        code: Instrument code — commodity code (``BRENT_CRUDE_USD``) or ticker.
        price: Numeric price; skipped if None (nothing worth persisting).
        currency: e.g. ``"USD"``.
        unit: e.g. ``"barrel"``.
        source: ``"oilprice"`` or ``"fmp"``.
        run_id: Override the current-context run id.
    """
    if not enabled() or price is None:
        return
    rid = run_id or current_run_id()
    if rid is None:
        return  # a price with no run to hang it on; skip rather than orphan
    try:
        price = float(price)
    except (TypeError, ValueError):
        return
    _execute(
        "INSERT INTO prices (run_id, code, price, currency, unit, source) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (rid, code, price, currency, unit, source),
    )


def save_response(agent: str, subject: Optional[str] = None, text: str = "",
                  rating: Optional[str] = None,
                  run_id: Optional[str] = None) -> None:
    """Record an agent's narrative output against a run (no-op if off).

    ``embedding`` is left NULL — a later pass can backfill vectors for semantic
    recall (see resumefromdb.md).

    Args:
        agent: Which agent produced the text, e.g. ``"coordinator"``.
        subject: Ticker / commodity the note is about.
        text: The narrative output. Skipped if empty.
        rating: ``BUY`` / ``HOLD`` / ``SELL`` if present, else None.
        run_id: Override the current-context run id.
    """
    if not enabled() or not text:
        return
    rid = run_id or current_run_id()
    _execute(
        "INSERT INTO agent_responses (run_id, agent, subject, text, rating) "
        "VALUES (%s, %s, %s, %s, %s)",
        (rid, agent, subject, text, rating),
    )
