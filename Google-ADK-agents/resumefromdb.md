# Resume from DB — A2A storage context

Session-handoff notes: the dedicated database for the **A2A agent system**
(`Google-ADK-agents/`). Created 2026-07-12. Pick up here in a new session.

---

## The database (Supabase / Postgres)

| Field | Value |
|-------|-------|
| Project name | `a2a-agents` |
| Project ref / id | `ketwfywvgzpvhawzcrvi` |
| Organization | `ElenaViewSynthesis's Org` (`aubskfbkerdwvqrenici`) |
| Region | `eu-west-2` (London) |
| Cost | $0 / month (free tier) |
| Postgres | v17 |
| API URL | `https://ketwfywvgzpvhawzcrvi.supabase.co` |

Dedicated to the A2A system — **separate** from the older `RISKagent` project
(`phmzdhbyliiqtwrltqmo`, INACTIVE), which belongs to `Equity-Research-agent/`
and is unrelated (that one only uses Supabase's S3 file storage, not the DB).

---

## Schema (applied — migration `init_a2a_storage`)

```
agent_runs ──┬──< prices            (1 run → many price rows)
             └──< agent_responses   (1 run → narrative output)
```

**`agent_runs`** — one row per agent invocation (the envelope):
`id (uuid), agent, subject, prompt, created_at`

**`prices`** — structured numeric **time-series** (raw tool results):
`id, run_id, code, price, currency, unit, source ('oilprice'|'fmp'), ts`
Indexes: `(code, ts desc)`, `(run_id)`.
Example: `BRENT_CRUDE_USD | 75.22 | USD | barrel | oilprice | 2026-07-12T13:30`

**`agent_responses`** — unstructured LLM **narrative** (verdicts / notes):
`id, run_id, agent, subject, text, rating (BUY/HOLD/SELL|null), embedding vector(1536), created_at`
Index: `(subject, created_at desc)`.

Rule of thumb: **numbers → `prices`, words → `agent_responses`, `agent_runs` ties them together.**

---

## How pgvector fits

- `pgvector` extension is **enabled**; it adds only one column: `agent_responses.embedding vector(1536)`.
- **Currently NULL / unused** — the table works as a plain text store today.
- Later: embed each response's `text` (→ 1536-dim vector), store in `embedding`,
  then do **similarity search** (`ORDER BY embedding <=> query_vec`) for "find
  past notes similar in meaning" / RAG (feed prior analyses back to an agent).
- **pgvector is for the *words*, not the *prices*.** Prices are exact numbers →
  normal SQL. Narrative has meaning → vectors. All in **one** Postgres DB; no
  separate vector store needed.

---

## Current state — persistence WIRED (2026-07-13)

- ✅ Project created, schema applied, `pgvector` enabled.
- ✅ **Write path: Supabase REST (PostgREST) with the service-role secret key**
  (`sb_secret_…`), which bypasses RLS. Chosen over direct Postgres because
  `psycopg`/`asyncpg`/`psql` are unavailable in the Linux run environment; the
  REST path needs only `httpx` (already a dependency).
- ✅ `SUPABASE_URL` + `SUPABASE_SECRET_KEY` live in `finance_coordinator/.env`
  (gitignored). `storage.py` also has a self-contained `.env` autoloader, so
  `enabled()` works even from a bare REPL / half-built venv.
- ✅ **Verified live**: full envelope (`agent_runs` → `prices` + `agent_responses`)
  writes correctly; test rows cleaned up (tables back to empty).

## The storage module (`a2a_finance/storage.py`)

- API: `enabled()`, `start_run(agent, subject, prompt) -> run_id`,
  `save_price(code, price, currency, unit, source)`, `save_response(agent,
  subject, text, rating)`, `current_run_id()`.
- **Graceful**: no-op if `SUPABASE_URL`/`SUPABASE_SECRET_KEY` unset, on any HTTP
  error; disables itself on 401/403 to avoid retry-storming a bad key.
- **Run scoping**: a `ContextVar`, plus a process-global fallback, plus an
  `A2A_RUN_ID` env fallback — so a run opened in one process propagates to the
  A2A specialist **subprocesses** (they inherit `A2A_RUN_ID` + `SUPABASE_*`).

## Wired producers

- `commodities_agent` `get_commodity_price` → `save_price(source=oilprice)`.
- `finance_coordinator` `get_stock_quote` → `save_price(source=fmp)`.
- `a2a_finance/run_demo.py`: `start_run` before booting services (exports
  `A2A_RUN_ID`), `save_response` on the coordinator's final note.
- `commodities_agent/run.py` (standalone, in-process): `start_run` +
  `save_response`; prices captured via the in-process run.

## Next steps

1. (Later) add embeddings to `agent_responses.embedding` for semantic recall
   (pgvector is enabled; column is NULL today).
2. Consider per-request `start_run` inside the A2A **services** themselves so
   direct service calls (not just via the demo driver) persist.

## Handy facts

- Persist decision (confirmed): store **both** prices and responses.
- MCP Supabase tools available: `apply_migration`, `execute_sql`, `list_tables`,
  `get_advisors`, `get_project_url`, `get_publishable_keys` (project id above).
- `get_advisors(type="security")` will flag RLS-with-no-policy — intentional for
  the server-only (service-role) write model.
