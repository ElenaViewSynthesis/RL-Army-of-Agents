#!/usr/bin/env python3
"""Backfill embeddings for agent_responses rows where embedding IS NULL.

One-off, **idempotent** (only touches NULL rows — safe to rerun). Embeds each
note's text with the local model (document mode) and PATCHes the vector back via
the Supabase REST API using the service-role key.

    uv run --extra recall python scripts/backfill_embeddings.py

Requires the ``recall`` extra installed (the embedding model) and the
SUPABASE_URL / SUPABASE_SECRET_KEY env (from finance_coordinator/.env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # scripts/ -> Google-ADK-agents/


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / "finance_coordinator" / ".env")
    sys.path.insert(0, str(PROJECT))

    import httpx

    from a2a_finance.embeddings import available, embed, to_pgvector

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL / SUPABASE_SECRET_KEY not set "
              "(finance_coordinator/.env)", file=sys.stderr)
        return 1
    if not available():
        print("Error: embedding model unavailable — install the 'recall' extra "
              "(uv sync --extra recall).", file=sys.stderr)
        return 1

    base = url.rstrip("/")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    # Fetch rows still missing an embedding.
    resp = httpx.get(
        f"{base}/rest/v1/agent_responses",
        params={"select": "id,text", "embedding": "is.null"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    print(f"{len(rows)} row(s) to embed.")

    embedded = 0
    for row in rows:
        vec = embed(row.get("text") or "", is_query=False)
        if vec is None:
            print(f"  skip id={row['id']} (embed failed / empty text)", file=sys.stderr)
            continue
        pr = httpx.patch(
            f"{base}/rest/v1/agent_responses",
            params={"id": f"eq.{row['id']}"},
            headers={**headers, "Prefer": "return=minimal"},
            json={"embedding": to_pgvector(vec)},
            timeout=30,
        )
        pr.raise_for_status()
        embedded += 1
        print(f"  embedded id={row['id']}")

    print(f"done: {embedded}/{len(rows)} embedded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
