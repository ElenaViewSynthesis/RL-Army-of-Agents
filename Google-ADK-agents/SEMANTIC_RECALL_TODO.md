# Semantic recall — TODO (zero-budget, local embeddings)

Goal: let agents **retrieve semantically similar past notes** from
`agent_responses` (pgvector), so prior analyses feed the next answer (RAG).

**Constraint: zero budget, fully local.** No OpenAI / Gemini / paid embedding
API. Embeddings come from a **local sentence-transformer** running on CPU.

Continues the persistence work in [`resumefromdb.md`](resumefromdb.md).
DB: Supabase project `a2a-agents` (ref `ketwfywvgzpvhawzcrvi`). Write path =
Supabase REST + service-role key (see `a2a_finance/storage.py`).

---

## Decision: model + dimensions  (revised — 1536, no migration)

- **Model:** `Alibaba-NLP/gte-Qwen2-1.5B-instruct` — outputs **1536-dim**
  embeddings **natively**, so it fits the existing `embedding vector(1536)`
  column with **no column migration**. Loaded with `trust_remote_code=True`.
- **Why this over MiniLM/BGE-small (384-dim):** those would force an
  `ALTER COLUMN … vector(384)` migration; gte-Qwen2 keeps the schema as-is.
  Trade-off: gte-Qwen2 is a **1.5B-param model** — much heavier: slower CPU
  inference and a **large first download** (multiple GB) vs MiniLM's ~80 MB.
  (MiniLM was install-tested and works — kept as a fallback if gte-Qwen2's
  weight/CPU cost proves impractical; that path would re-add the 384 migration.)
- **The SAME model must embed both stored notes and query text** — never mix
  models, or distances become meaningless.
- **Instruct-model query asymmetry (important):** GTE *instruct* models expect a
  task **instruction prefix on the QUERY** but embed **documents plainly**. So
  `embed()` needs a `is_query` mode: documents → `model.encode(text)`; queries →
  `model.encode(text, prompt_name="query")` (or manually prepend
  `"Instruct: Given a search query, retrieve relevant notes\nQuery: "`). Getting
  this wrong quietly degrades recall quality.
- **Hard dimension check in code:** `if len(vec) != 1536: return None` — never
  write a wrong-width vector to a `vector(1536)` column.

---

## Steps

### 1. HNSW index — SAFE, no code/venv needed (apply now)
```sql
create index if not exists agent_responses_embedding_hnsw
  on agent_responses using hnsw (embedding vector_cosine_ops);
```
Cosine ops pair with the `<=>` operator in the RPC. (pgvector HNSW supports up to
2000 dims; 1536 is fine. Table is ~empty, so the build is instant.)

### 2. The similarity-search RPC — SAFE, no code/venv needed (apply now)
```sql
create or replace function match_responses(
  query_embedding vector(1536),
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

### 3. Add the dependency (optional extra + CPU torch)
- Add as an **optional** extra so torch isn't forced on the core install:
  ```toml
  [project.optional-dependencies]
  recall = ["sentence-transformers>=3.0.0"]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }
  ```
- Install with `uv sync --extra recall`. Core install stays light.
- **Install-test gte-Qwen2 in WSL first** (isolated venv, like the MiniLM test) —
  it's much bigger; confirm it loads on CPU and returns `dim: 1536` before wiring.

### 4. `embed()` helper  (a2a_finance/embeddings.py)
- Lazy-load the model **once** into a module-global (loading is slow; reuse it):
  ```python
  SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                      trust_remote_code=True)   # pin a revision before prod
  ```
- `embed(text, is_query=False) -> list[float] | None` — documents plain, queries
  with the instruction prompt (see the query-asymmetry note above).
  `normalize_embeddings=True`. **Hard-check** `len(vec) == 1536` else return None.
- `to_pgvector(vec) -> str` → `"[" + ",".join(map(repr, vec)) + "]"` (pgvector
  wants a string literal over REST; PostgREST casts text → vector).
- Import `sentence_transformers` lazily inside the function so a missing extra
  just disables recall (returns None) without breaking price/response writes.

### 5. Embed on write
- Extend `storage.save_response(...)`: compute `embed(text)` (document mode);
  if not None, include `"embedding": to_pgvector(vec)` in the POST body.
- Best-effort: on failure still insert with `embedding = NULL` (Step 6 backfills).

### 6. Backfill existing NULL rows (one-off script)
- `scripts/backfill_embeddings.py`: page `SELECT id, text FROM agent_responses
  WHERE embedding IS NULL`, `embed()` each (document mode), `PATCH
  /rest/v1/agent_responses?id=eq.<id>` with `{"embedding": "[…]"}`. Idempotent.
  Today that's the single WTI row.

### 7. `search_similar()`  (a2a_finance/storage.py)
- `search_similar(query_text, subject=None, limit=5) -> list[dict]`:
  `embed(query_text, is_query=True)` → `POST /rest/v1/rpc/match_responses` with
  `{"query_embedding": to_pgvector(vec), "match_count": limit,
    "filter_subject": subject}`. No-op (`[]`) if persistence/embeddings off.

### 8. Expose as an ADK tool + wire into an agent
- Tool `recall_similar_notes(query, subject="")` wrapping `search_similar`.
- Add it to the **research agent** (`a2a_finance/research.py`) and/or coordinator;
  instruction: *"Before writing your note, call `recall_similar_notes` and
  reconcile with any prior analyses."*

### 9. Verify
- Embed + insert 2–3 toy notes on the same ticker; confirm `match_responses`
  ranks the semantically closest first (similarity descending).
- Confirm graceful degradation: without the `recall` extra, price + response
  writes still work; recall returns `[]`. Clean up toy rows.

---

## Notes / gotchas
- **Dimension must be 1536 everywhere:** model output, the `len(vec)` check, the
  column, and the RPC signature.
- **`trust_remote_code=True` runs code from the model repo** — pin the model name
  AND a specific revision (commit) before treating it as production.
- **Instruct query prefix** (Step Decision) is easy to miss and silently hurts
  recall — documents plain, queries instructed.
- **First-call latency:** large weight download (first run) + slower CPU encode
  than MiniLM. Load lazily so importing `storage` stays cheap.
- **RLS:** the RPC is called with the service-role key → bypasses RLS.
- **pgvector over REST:** always pass vectors as the string literal `"[…]"`.
