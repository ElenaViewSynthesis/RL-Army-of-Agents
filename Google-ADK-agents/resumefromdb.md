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

## Current state / where we stopped

- ✅ Project created, schema applied, `pgvector` enabled.
- ⚠️ Tables are **empty** — the agents do **not** write to them yet.
- ⚠️ **RLS is enabled with no policies** on all three tables → only a
  service-role key or a direct Postgres connection (trusted, server-side) can
  write. Anon/publishable keys are blocked by design.
- 🔑 **No secret was fetched** (key retrieval was declined). Nothing in `.env`
  yet for this DB.

---

## Next steps (to wire persistence)

1. **Decide the write path** (server-side, bypasses RLS):
   - **Direct Postgres** (recommended): `psycopg` + connection string
     `postgresql://postgres:<PASSWORD>@db.ketwfywvgzpvhawzcrvi.supabase.co:5432/postgres`
     — get the password from Supabase dashboard → Project Settings → Database.
   - **or** Supabase REST (PostgREST) + service-role key.
2. Add the chosen secret to `finance_coordinator/.env` (gitignored) —
   e.g. `A2A_DB_URL=postgresql://…`.
3. Build a **storage module** (`a2a_finance/storage.py`): `start_run(...)`,
   `save_price(...)`, `save_response(...)` — no-ops gracefully if the env var
   is unset.
4. Wire it in: agents/tools call `save_price` on each OilPrice/FMP fetch and
   the coordinator/specialists call `save_response` with their final text.
5. (Later) add embeddings to `agent_responses.embedding` for semantic recall.
6. Add `psycopg[binary]` (or `supabase`) to `pyproject.toml`.

## Handy facts

- Persist decision (confirmed): store **both** prices and responses.
- MCP Supabase tools available: `apply_migration`, `execute_sql`, `list_tables`,
  `get_advisors`, `get_project_url`, `get_publishable_keys` (project id above).
- Run advisors after wiring: `get_advisors(type="security")` — RLS-with-no-policy
  will flag; that's intentional for a server-only write model, but review.
