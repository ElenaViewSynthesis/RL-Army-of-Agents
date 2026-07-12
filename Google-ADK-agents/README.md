# Google ADK Agents

Multi-agent financial-research orchestrator built on **Google's Agent Development Kit (ADK 2.3)**. A coordinator agent routes each request to the right specialist via ADK's LLM-driven delegation (`sub_agents`), then synthesizes the results.

Two selectable agents ship in this project:

## 1. `finance_coordinator` — interactive router (multi-model)

```
finance_coordinator          (root_agent — pure router, no data tools)   [Gemini]
├── fundamentals_agent       profile · quote · TTM key metrics           [Gemini]
├── valuation_agent          DCF fair value · peers · analyst consensus  [OpenRouter]
└── risk_agent               leverage · margin fragility · red flags     [Gemini]
```

The coordinator holds no market tools itself — it delegates to the specialist whose `description` best matches the query. Best for ad-hoc questions ("is TSLA expensive?", "risks in AAPL?").

**Heterogeneous models, one tree.** The coordinator and two specialists run on **Gemini** directly; the `valuation_agent` runs on an **open model via OpenRouter** (LiteLLM — default `meta-llama/llama-3.3-70b-instruct`). ADK's delegation is model-agnostic, so a Gemini coordinator can transfer to an OpenRouter-backed sub-agent and back. Model wiring lives in `config.py` (`GEMINI_MODEL` string vs `openrouter_model()` LiteLLM handle); swap either via `ADK_MODEL` / `OPENROUTER_MODEL` in `.env`.

## 2. `equity_report_pipeline` — deterministic full report

```
equity_report_pipeline       (SequentialAgent — fixed order, shared state)
├── fundamentals_step   ─┐
├── valuation_step       ├─ each writes its section to state via output_key
├── risk_step           ─┘
└── synthesis_step       reads {fundamentals_section}/{valuation_section}/
                         {risk_section} → rated note (BUY/HOLD/SELL + target)
```

A `SequentialAgent` runs the three analysts in a fixed order — no LLM routing — each persisting its section to session state. The tool-less synthesis step (`include_contents="none"`) pulls the sections back through instruction templating and emits one institutional research note. Best for "give me the full report on NVDA".

Both agents share the same tools and model config. Market tools call the **live Financial Modeling Prep (FMP) `/stable` API** (6 core endpoints: profile, quote, key-metrics-ttm, discounted-cash-flow, grades + price-target-consensus, stock-peers). Tools return `{"error": ...}` on a missing key / premium gate / network failure so the model reports the gap rather than crashing.

## Layout

```
Google-ADK-agents/
├── pyproject.toml               # uv-managed; google-adk>=2.3.0
├── finance_coordinator/
│   ├── __init__.py              # exposes agent.root_agent to ADK
│   ├── agent.py                 # root_agent = coordinator + sub_agents
│   ├── config.py                # env-driven model selection (ADK_MODEL)
│   ├── sub_agents/              # fundamentals / valuation / risk specialists
│   └── tools/
│       ├── fmp_client.py        # shared FMP /stable GET helper
│       └── market_tools.py      # 6 live FMP tool functions
├── report_pipeline/
│   ├── __init__.py              # exposes agent.root_agent to ADK
│   ├── agent.py                 # root_agent = SequentialAgent(steps)
│   └── steps.py                 # 4 steps; reuses finance_coordinator tools
└── .env.example
```

## Setup

```bash
# from Google-ADK-agents/
uv sync                          # install google-adk into .venv

# ADK reads the .env next to the agent package
cp .env.example finance_coordinator/.env
# edit finance_coordinator/.env → set:
#   GEMINI_API_KEY      (Gemini agents)
#   OPENROUTER_API_KEY  (OpenRouter-backed agents — A2A specialists, commodities)
#   FMP_API_KEY         (equity market-data tools)
#   OILPRICE_API_KEY    (commodities agent)
```

> **Busy-traffic timeout:** OpenRouter-backed calls default to a **10-minute** per-request timeout (`OpenRouterLlm.timeout_ms`), since free/queued models can sit in a queue for minutes under load. Override with `OPENROUTER_TIMEOUT_MS`.

## Run

```bash
uv run adk web                          # browser dev UI — pick either agent from the dropdown

uv run adk run finance_coordinator      # interactive router (CLI)
uv run adk run report_pipeline          # full-report pipeline (CLI)
```

- Router — try: `is TSLA expensive right now?`, `what are the risks in AAPL?`
- Pipeline — try: `NVDA` or `full report on MSFT` → runs all four steps → rated note.

> `adk run`/`adk web` target a package `root_agent` (both Gemini), so they need Gemini quota. To exercise the **OpenRouter reasoning agent without Gemini**, use the runner below.

### `run_valuation.py` — stream the reasoning agent (no Gemini needed)

Drives the OpenRouter reasoning `valuation_agent` directly. **Streaming is the default**; `--structured` emits a Pydantic-validated `ResearchNote` as JSON via a two-step flow (reasoning agent gathers with tools → non-reasoning formatter agent with `output_schema` emits the note).

```bash
uv run python run_valuation.py NVDA                    # stream tokens (default)
uv run python run_valuation.py AAPL -q "cheap vs DCF?"  # streaming, specific question
uv run python run_valuation.py TSLA --structured         # schema-validated JSON note
uv run python run_valuation.py NVDA --no-reasoning -m meta-llama/llama-3.3-70b-instruct
```

`schema.py` holds the `ResearchNote` Pydantic model (the Python analogue of the TS zod schema). Runs from any working directory (WSL-friendly) — paths resolve relative to the script.

---

## Running the agents

All OpenRouter-backed agents run **without Gemini**. See [`SHOWCASE.md`](SHOWCASE.md) for the full prompt catalog.

### 🛢️ Commodities agent (OilPrice API)

Live oil/gas/metal/coal prices, history, and the curated Fuse Energy watchlist. Standalone, streaming:

```bash
# from Google-ADK-agents/
uv run python -m commodities_agent.run "Show me the Fuse Energy watchlist prices."
uv run python -m commodities_agent.run "What is WTI crude trading at, and how has it moved this month?"

# "daily glimpse" — all endpoints at once (priority: gasoil, UK gas, petroleum):
uv run python -m commodities_agent.run \
  "Show the Fuse Energy watchlist prices, then report the past-day movement for GASOIL_USD, NATURAL_GAS_GBP, and BRENT_CRUDE_USD."
```

See [`commodities_agent/`](commodities_agent/) — [`OILPRICE_API.md`](commodities_agent/OILPRICE_API.md) (provider ref), [`FUSE_ENERGY_WATCHLIST.md`](commodities_agent/FUSE_ENERGY_WATCHLIST.md) (curated codes).

### 🔀 A2A multi-agent system

Specialists run as independent A2A services (`:8001` valuation · `:8002` fundamentals · `:8003` risk · `:8004` commodities); a coordinator delegates to them over HTTP. `run_demo.py` boots the services, runs the coordinator, then tears down:

```bash
# from Google-ADK-agents/ — routing mode (one query → the matching specialist)
uv run python a2a_finance/run_demo.py AAPL "what are the key risks?"
uv run python a2a_finance/run_demo.py WTI  "current price of WTI crude oil?"   # → commodities service

# research fan-out — consult all equity specialists, synthesize one note
# (override the slow reasoning defaults for a fast run):
A2A_VALUATION_MODEL=meta-llama/llama-3.3-70b-instruct \
A2A_RISK_MODEL=meta-llama/llama-3.3-70b-instruct \
  uv run python a2a_finance/run_demo.py NVDA --research

# or run one service manually and inspect its agent card:
uv run uvicorn a2a_finance.commodities_service:a2a_app --port 8004
curl http://localhost:8004/.well-known/agent-card.json
```

Full details in [`a2a_finance/README.md`](a2a_finance/README.md).

### 🟦 Cross-runtime TS node (Tier B)

The Python coordinator can delegate to a **TypeScript** agent (`@openrouter/agent` wrapped in `@a2a-js/sdk`) over A2A. Start the TS service, then route to it:

```bash
# terminal 1 — from ../OpenRouter-Agent/
npm run a2a                      # → http://localhost:8100  (agent card at /.well-known/agent-card.json)

# terminal 2 — from Google-ADK-agents/
uv run python a2a_finance/run_demo.py MSFT "give me a general read"   # coordinator routes to the TS agent
```

Because `RemoteA2aAgent` resolves cards lazily, the coordinator still runs with only the Python specialists if the TS service is down — it just can't route there.

## Next steps

- Add the premium FMP endpoints (SEC filings, ETF/mutual-fund holdings, ownership) as further tools.
- Wire the pipeline into a `Runner` + session service for programmatic/API use.
