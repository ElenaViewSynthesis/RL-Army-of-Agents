# Containerization & Kubernetes — implementation plan (resume doc)

Session-handoff plan for Dockerizing the ADK finance agents and running them on
Kubernetes. Pick up here in a new session. Nothing below is implemented yet —
this is the plan only.

Context: the app is `Google-ADK-agents/` — a multi-agent LLM system (Google ADK
+ A2A protocol) with OpenRouter/Gemini models, external SaaS state (Supabase,
TimescaleDB/Tiger Cloud, Langfuse), and 3 batch seed cron jobs. All commits are
scoped to `Google-ADK-agents/`.

---

## Time estimate

| Target | Realistic effort |
|--------|------------------|
| **Docker + docker-compose** working locally (4 A2A services talking to each other) | **1–2 days** |
| **+ Basic Kubernetes** (local `kind`/`k3d`, Deployments + Services + CronJobs) | **+1–2 days** → ~3–4 days total |
| **Production-grade** (managed GKE/EKS, ingress/TLS, secrets mgmt, CI/CD, autoscaling, coordinator-as-service) | **~1.5–2 weeks** total |

Big variables: **dev cluster vs. managed prod**, and whether the **coordinator is
exposed as a queryable API** (a real refactor) or just specialists + batch jobs.

---

## Why it's faster than usual (project-specific)

- **No databases to run in-cluster** — Supabase, TimescaleDB (Tiger Cloud), and
  Langfuse are all **SaaS**. Eliminates StatefulSets, PVCs, backups, DB operators
  (usually the hardest K8s part).
- **Service discovery is already env-driven** — the coordinator's `A2A_*_CARD` /
  `A2A_*_PORT` env vars point at K8s service DNS (`http://fundamentals-svc:8002`)
  with **zero code changes**.
- **Graceful degradation everywhere** — pods won't crash-loop on a missing
  optional dep/key (storage, embeddings, tiger, observability all no-op).
- **Cron jobs map 1:1 to K8s CronJobs** (the 3 seed scripts) — replacing the WSL
  crontab cleanly.
- **Natural probe endpoint** — each service's `/.well-known/agent-card.json` is a
  ready-made readiness probe.

---

## Implementation plan (phased)

### Phase 1 — Dockerize (~1 day)
- Multi-stage `Dockerfile`: `python:3.12-slim` → install `uv` →
  `uv sync --frozen --extra timescale --extra observability` (skip
  `recall`/torch — multi-GB; separate image only if needed) → copy source →
  non-root user.
- **One image, many entrypoints** (each A2A service = same image, different `CMD`).
- `.dockerignore` (`.venv`, `.git`, `.agents`, `__pycache__`, `*.pyc`).

### Phase 2 — docker-compose (~0.5 day)
- Compose file for the 4 services (+ the TS node from `../OpenRouter-Agent` for
  Tier B if wanted), `env_file: finance_coordinator/.env`, `A2A_*_CARD` pointed at
  compose service names. Validate a coordinator query end-to-end.

### Phase 3 — Kubernetes core (~1–2 days)
- **Deployments + ClusterIP Services** for each A2A specialist
  (`fundamentals`/`valuation`/`risk`/`commodities`), optionally `adk web`.
- **ConfigMap** (ports, models, `A2A_*_CARD` → service DNS) + **Secret** (API
  keys, DB creds, Langfuse keys — from `.env`).
- **Readiness/liveness probes** on the agent-card path; resource requests/limits.
- **CronJobs** for `seed_timescale_prices` / `seed_insider_trades` /
  `seed_marine_ports` (single replica each, to respect the 200/day OilPrice limit).

### Phase 4 — Coordinator as a service (optional, ~0.5–1 day)
- Today `run_demo.py` is a CLI that spawns subprocesses over `localhost`. For K8s,
  wrap the coordinator in a small FastAPI/uvicorn app (a query endpoint) that talks
  to the specialist **Services** via DNS. Skip if only specialists + batch jobs are
  needed.

### Phase 5 — CI/CD + registry (~0.5–1 day)
- Build/push image to GHCR/ECR via GitHub Actions on commit; K8s pulls the tagged
  image.

### Phase 6 — Prod hardening (managed clusters, ~2–5 days)
- Managed cluster, Ingress + TLS, **secrets management** (sealed-secrets /
  external-secrets — important, `.env` has many live keys), HPA/autoscaling, and
  cluster logs/metrics (Langfuse already covers the LLM-trace layer).

---

## Project-specific gotchas to plan for

- **Coordinator refactor** — the `localhost:800X` subprocess model → K8s service
  DNS (env already supports it, but you need a persistent coordinator, not the CLI).
- **Image size** — keep `torch`/`sentence-transformers` (`recall` extra) out of
  the default image; it's huge. Separate image/job if semantic recall is needed.
- **Rate limits** — seed CronJobs must stay **single-replica** (OilPrice 200/day;
  no parallel dupes). Upserts are idempotent, so a missed run self-heals.
- **Slow reasoning models** — OpenRouter free-tier queues minutes → generous
  probe/timeout tolerances; autoscaling won't fix provider queueing.
- **Secrets** — never bake `.env` into the image; use K8s Secrets, and rotate the
  keys that were shared in chat (Supabase service key, Langfuse secret, Tiger DB
  password).

---

## Current-state facts useful for this work

- **Ports:** fundamentals `:8002`, valuation `:8001`, risk `:8003`, commodities
  `:8004`, TS node `:8100`. Overridable via `A2A_*_PORT` / `A2A_*_CARD` env.
- **Entry points:** `a2a_finance/{fundamentals,valuation,risk,commodities}_service.py`
  (uvicorn ASGI apps `a2a_app`); discoverable agents `finance_coordinator`,
  `commodities_agent`, `energy_drilling_agent`, `report_pipeline`; drivers
  `run_demo.py`, `run_valuation.py`, `*/run.py`.
- **Optional extras:** `timescale` (psycopg), `recall` (sentence-transformers +
  CPU torch), `observability` (openinference + otel exporter).
- **Seed scripts + crons:** `seed_timescale_prices.py` (daily),
  `seed_insider_trades.py` (30-min market hours), `seed_marine_ports.py` (weekly);
  wrappers in `scripts/*_cron.sh` (currently WSL crontab → become K8s CronJobs).
- **Env file:** `finance_coordinator/.env` (gitignored) holds all secrets:
  OPENROUTER/FMP/OILPRICE/GEMINI keys, SUPABASE_*, TIGER_DATABASE_*, LANGFUSE_*.
- **Tracing:** `a2a_finance/observability.py` — init before `google.adk` import;
  OTLP → Langfuse. All 8 agents wired.

---

## Recommended first step when resuming

Start **Phase 1**: write `Dockerfile`, `.dockerignore`, and a `docker-compose.yml`
for the four A2A services; verify it builds and a coordinator query works locally
before touching Kubernetes.
