"""Optional persistence for the A2A finance system (Supabase / Postgres).

Writes to the dedicated Supabase project (``a2a-agents``) documented in
``resumefromdb.md``. Three tables, one envelope::

    agent_runs ──┬──< prices            (numbers: raw tool results)
                 └──< agent_responses   (words: LLM narrative / verdicts)

**Write path: Supabase REST (PostgREST) with the service-role secret key.**
RLS is enabled with no policies, so the publishable (anon) key is blocked by
design; the ``sb_secret_…`` key maps to ``service_role`` and bypasses RLS. Only
``httpx`` is needed (already a dependency) — no direct Postgres driver. Config::

    SUPABASE_URL=https://ketwfywvgzpvhawzcrvi.supabase.co
    SUPABASE_SECRET_KEY=sb_secret_…        # service-role; keep gitignored

**Graceful by design.** If either env var is unset, or a request fails, every
function is a silent no-op — the agents run exactly as before, just without
persistence. Persistence is a side effect; it must never break a run. An auth
failure (401/403) disables persistence for the process so a misconfigured key
doesn't retry-storm.

Run-scoping uses a :class:`~contextvars.ContextVar`: ``start_run`` opens the
envelope and records its id on the context, and ``save_price`` / ``save_response``
attach to the current run without the caller threading an id through every tool
signature. The context var is per-process — in the multi-process A2A routing path
a run opened in the coordinator process is not visible inside a separate
specialist service process (each process opens its own run if it calls
``start_run``).
"""

from __future__ import annotations

import contextvars
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("a2a_finance.storage")

# finance_coordinator/.env holds the Supabase secrets. Run drivers load it via
# python-dotenv, but a bare ``import storage`` (or a REPL check of enabled())
# wouldn't — so fall back to a tiny self-contained parser. No dependency on
# python-dotenv, so this survives a half-built venv. Only fills vars that are
# absent; an explicit environment always wins.
_ENV_PATH = Path(__file__).resolve().parent.parent / "finance_coordinator" / ".env"
_env_loaded = False


def _ensure_env_loaded() -> None:
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"):
        return  # already in the environment; nothing to do
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
        pass  # no .env here; rely on the ambient environment

# Current run id for this execution context; save_* default to it.
_run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "a2a_run_id", default=None
)
# Process-global fallback: ADK may dispatch sync tools to a thread pool without
# copying the context, so the ContextVar can be invisible inside a tool. The CLI
# drivers run exactly one run per process, so a module global is a safe fallback.
# (Don't rely on it in a long-lived multi-run server — there the ContextVar, set
# per request, is the correct scope.)
_process_run_id: Optional[str] = None

_TIMEOUT = 10.0  # best-effort; never stall an agent turn on the DB
_disabled = False  # set on an auth failure so we don't retry-storm a bad key


def _config() -> Optional[tuple[str, str]]:
    """Return ``(base_url, secret_key)`` if persistence is configured, else None."""
    _ensure_env_loaded()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        return None
    return url.rstrip("/"), key


def enabled() -> bool:
    """True if persistence is configured (``SUPABASE_URL`` + ``SUPABASE_SECRET_KEY``)."""
    return not _disabled and _config() is not None


def _post(table: str, row: dict, *, return_row: bool = False) -> Optional[dict]:
    """POST one row to a PostgREST table. Returns the inserted row if requested.

    Never raises: any failure logs a warning and returns None. A 401/403 disables
    persistence for the rest of the process (bad/expired key).
    """
    global _disabled
    cfg = _config()
    if _disabled or cfg is None:
        return None
    base, key = cfg
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if return_row else "return=minimal",
    }
    try:
        resp = httpx.post(
            f"{base}/rest/v1/{table}", json=row, headers=headers, timeout=_TIMEOUT
        )
        if resp.status_code in (401, 403):
            logger.warning(
                "A2A persistence auth failed (%s) — disabling; check "
                "SUPABASE_SECRET_KEY.", resp.status_code,
            )
            _disabled = True
            return None
        resp.raise_for_status()
        if return_row:
            data = resp.json()
            return data[0] if isinstance(data, list) and data else None
        return None
    except Exception as e:  # network, timeout, HTTP error, JSON decode
        logger.warning("A2A persistence write to %s failed (skipped): %s", table, e)
        return None


def _rpc(name: str, payload: dict) -> list:
    """Call a PostgREST RPC (`/rest/v1/rpc/<name>`) and return its rows.

    Never raises: returns [] on any failure. Used for the vector-search function
    (`match_responses`) that plain PostgREST queries can't express.
    """
    global _disabled
    cfg = _config()
    if _disabled or cfg is None:
        return []
    base, key = cfg
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            f"{base}/rest/v1/rpc/{name}", json=payload, headers=headers, timeout=_TIMEOUT
        )
        if resp.status_code in (401, 403):
            logger.warning("A2A persistence auth failed (%s) — disabling.", resp.status_code)
            _disabled = True
            return []
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("A2A rpc %s failed (skipped): %s", name, e)
        return []


def start_run(agent: str, subject: Optional[str] = None,
              prompt: Optional[str] = None) -> Optional[str]:
    """Open a run envelope and set it as the current run for this context.

    Args:
        agent: The agent/driver opening the run, e.g. ``"coordinator"``.
        subject: Ticker or commodity the run is about, e.g. ``"NVDA"``.
        prompt: The originating user prompt.

    Returns:
        The new run's uuid (str), or None when persistence is off / the write fails.
    """
    if not enabled():
        return None
    row = _post(
        "agent_runs",
        {"agent": agent, "subject": subject, "prompt": prompt},
        return_row=True,
    )
    run_id = str(row["id"]) if row and row.get("id") else None
    if run_id:
        global _process_run_id
        _run_id_var.set(run_id)
        _process_run_id = run_id
    return run_id


def current_run_id() -> Optional[str]:
    """The run id for the current context (ContextVar first, process global fallback)."""
    return _run_id_var.get() or _process_run_id or os.getenv("A2A_RUN_ID")


def save_price(code: str, price: Optional[float], currency: Optional[str] = None,
               unit: Optional[str] = None, source: Optional[str] = None,
               run_id: Optional[str] = None) -> None:
    """Record one price point against a run (no-op if persistence is off).

    Args:
        code: Instrument code — commodity code (``BRENT_CRUDE_USD``) or ticker.
        price: Numeric price; skipped if None or non-numeric.
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
    _post(
        "prices",
        {"run_id": rid, "code": code, "price": price,
         "currency": currency, "unit": unit, "source": source},
    )


def save_response(agent: str, subject: Optional[str] = None, text: str = "",
                  rating: Optional[str] = None,
                  run_id: Optional[str] = None) -> None:
    """Record an agent's narrative output against a run (no-op if off).

    Best-effort embeds ``text`` (document mode) into ``embedding`` for semantic
    recall. If the ``recall`` extra / model is unavailable the embedding is
    simply omitted (column stays NULL) and can be backfilled later — the note is
    always written regardless.

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
    if rid is None:
        return
    row = {"run_id": rid, "agent": agent, "subject": subject,
           "text": text, "rating": rating}
    # Embeddings default to the backfill path (scripts/backfill_embeddings.py) so a
    # heavy model never loads/infers on the note-write hot path. Opt into
    # synchronous embed-on-write with A2A_EMBED_ON_WRITE=1 (accepts the latency).
    if os.getenv("A2A_EMBED_ON_WRITE", "").lower() in ("1", "true", "yes"):
        try:
            from a2a_finance.embeddings import embed, to_pgvector

            vec = embed(text, is_query=False)
            if vec is not None:
                row["embedding"] = to_pgvector(vec)
        except Exception:
            pass
    _post("agent_responses", row)


def search_similar(query_text: str, subject: Optional[str] = None,
                   limit: int = 5) -> list:
    """Return past notes semantically similar to ``query_text`` (via pgvector).

    Embeds the query (query mode) and calls the ``match_responses`` RPC, which
    ranks stored notes by cosine similarity. Returns [] if persistence or the
    embedding model is unavailable, or nothing matches.

    Args:
        query_text: What to search for (ticker + question, or a theme).
        subject: Optional exact subject filter (e.g. ticker "NVDA").
        limit: Max matches to return.
    """
    if not enabled() or not query_text:
        return []
    try:
        from a2a_finance.embeddings import embed, to_pgvector
    except Exception:
        return []
    vec = embed(query_text, is_query=True)
    if vec is None:
        return []
    return _rpc(
        "match_responses",
        {"query_embedding": to_pgvector(vec),
         "match_count": limit,
         "filter_subject": subject or None},
    )
