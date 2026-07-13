# Semantic recall — TODO (zero-budget, local embeddings)

Goal: let agents **retrieve semantically similar past notes** from
`agent_responses` (pgvector), so prior analyses feed the next answer (RAG).

**Constraint: zero budget, fully local.** No OpenAI / Gemini / paid embedding
API. Embeddings come from a **local sentence-transformer** running on CPU.

Continues the persistence work in [`resumefromdb.md`](resumefromdb.md).
DB: Supabase project `a2a-agents` (ref `ketwfywvgzpvhawzcrvi`). Write path =
Supabase REST + service-role key (see `a2a_finance/storage.py`).

---

## Decision: model + dimensions

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~80 MB, CPU-only,
  fast, the standard zero-budget default. (Quality alt, also 384-dim:
  `BAAI/bge-small-en-v1.5`. Higher quality but bigger/slower, 768-dim:
  `all-mpnet-base-v2` — would need `vector(768)` instead.)
- **Column must match the model.** The schema ships `embedding vector(1536)`
  (an OpenAI-era default). MiniLM outputs **384**, so we re-dimension the column
  to `vector(384)`. Safe today: the only stored response (the WTI run) has
  `embedding IS NULL`, so no embeddings are lost.
- **The SAME model must embed both stored notes and query text** — never mix
  models, or distances become meaningless. If we ever switch models, re-embed all
  rows and re-dimension.
- **Dependency weight:** `sentence-transformers` pulls in `torch` (large, 100s of
  MB–GB, but free). Lighter alternative to evaluate if the install is painful:
  **`fastembed`** (Qdrant) — ONNX runtime, no torch, ships `bge-small-en-v1.5`
  (384-dim) — a near drop-in for `embed()`.

---

## Steps

### 0. Add the dependency
- Add `sentence-transformers` (or `fastembed`) to `pyproject.toml`, `uv sync`.
- First model load downloads weights to the HF cache (one-time, offline after).

### 1. Re-dimension the column (migration)
```sql
-- no vector index exists yet, and embeddings are all NULL, so this is clean
ALTER TABLE agent_responses ALTER COLUMN embedding TYPE vector(384);
```
Apply via Supabase MCP `apply_migration` (name e.g. `embedding_dim_384`).

### 2. `embed()` helper  (a2a_finance/embeddings.py)
- Lazy-load the model **once** into a module-global (loading is slow; reuse it).
- `embed(text: str) -> list[float] | None` — returns the 384-float vector; returns
  `None` on any failure (graceful, like the rest of storage).
- Add `to_pgvector(vec) -> str` → `"[" + ",".join(map(repr, vec)) + "]"` (pgvector
  wants a **string literal** over REST; PostgREST casts text → vector).
- Keep the import of `sentence_transformers` inside the function/lazy so a missing
  dep just disables embeddings (recall off) without breaking price/response writes.

### 3. HNSW index (migration)
```sql
CREATE INDEX IF NOT EXISTS agent_responses_embedding_hnsw
  ON agent_responses USING hnsw (embedding vector_cosine_ops);
```
Cosine ops pair with the `<=>` operator used in the RPC.

### 4. Embed on write
- Extend `storage.save_response(...)`: after computing `embed(text)`, include
  `"embedding": to_pgvector(vec)` in the POST body (only if `vec` is not None).
- Best-effort: if embedding fails, still insert the row with `embedding = NULL`
  (Step 5's backfill catches it later). Persistence must never fail on embeddings.

### 5. Backfill existing NULL rows (one-off script)
- `scripts/backfill_embeddings.py`: page `SELECT id, text FROM agent_responses
  WHERE embedding IS NULL` over REST, `embed()` each, `PATCH
  /rest/v1/agent_responses?id=eq.<id>` with `{"embedding": "[…]"}`.
- Idempotent (only touches NULLs). Today that's the single WTI row.

### 6. The similarity-search RPC (migration)
```sql
create or replace function match_responses(
  query_embedding vector(384),
  match_count int default 5,
  filter_subject text default null
)
returns table (id bigint, subject text, text text, rating text, similarity float)
language sql stable as $$
  select a.id, a.subject, a.text, a.rating,
         1 - (a.embedding <=> query_embedding) as similarity
  from agent_responses a
  where a.embedding is not null
    and (filter_subject is null or a.subject = filter_subject)
  order by a.embedding <=> query_embedding
  limit match_count;
$$;
```
Why an RPC: PostgREST has no syntax for `ORDER BY embedding <=> $q`, so the ANN
search lives in a SQL function, exposed at `POST /rest/v1/rpc/match_responses`.
`<=>` = cosine distance; `1 - distance` = similarity (1.0 = identical). The
`order by … <=>` is what the HNSW index accelerates.

### 7. `search_similar()`  (a2a_finance/storage.py)
- `search_similar(query_text, subject=None, limit=5) -> list[dict]`:
  `embed(query_text)` → `POST /rest/v1/rpc/match_responses` with
  `{"query_embedding": to_pgvector(vec), "match_count": limit,
    "filter_subject": subject}`. No-op (returns `[]`) if persistence/embeddings off.

### 8. Expose as an ADK tool + wire into an agent
- Tool `recall_similar_notes(query: str, subject: str = "") -> dict` wrapping
  `search_similar` (returns top-k prior notes + similarity).
- Add it to the **research agent** (`a2a_finance/research.py`) and/or coordinator.
- Update the instruction: *"Before writing your note, call `recall_similar_notes`
  and reconcile with any prior analyses; cite when a past note informs the view."*

### 9. Verify
- Embed + insert 2–3 toy notes on the same ticker; confirm `match_responses`
  ranks the semantically closest first (similarity descending).
- Confirm graceful degradation: with `sentence-transformers` uninstalled, price +
  response writes still work; recall simply returns `[]`.
- Clean up toy rows.

---

## Notes / gotchas
- **Dimension mismatch = hard error.** Column, model output, and RPC signature must
  all be 384 (or all whatever the chosen model emits).
- **Normalize?** MiniLM outputs are already ~unit-norm; cosine distance is the
  right metric. Use `normalize_embeddings=True` in `model.encode(...)` to be safe.
- **First call latency:** model load (seconds) + weight download (first run only).
  Load lazily so importing `storage` stays cheap.
- **RLS:** the RPC is called with the service-role key → bypasses RLS like the rest.
- **pgvector over REST:** always pass vectors as the string literal `"[…]"`.
