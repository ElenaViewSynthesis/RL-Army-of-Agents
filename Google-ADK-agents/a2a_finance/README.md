# A2A Finance — inter-agent coordination over the A2A protocol

Tier-A [Agent2Agent (A2A)](https://google.github.io/A2A/) coordination for the finance agents. Specialist agents run as **independent A2A services**; a coordinator delegates to them **over HTTP** via `RemoteA2aAgent` — not in-process `sub_agents`. Everything runs on the **OpenRouter client SDK** (`OpenRouterLlm`), so no Gemini is needed.

```
finance_coordinator (OpenRouterLlm)              ── coordinator process (Python) ──
        │  transfer_to_agent  ──▶  A2A / HTTP
        ├──▶ fundamentals_agent   (:8002)   profile · quote · TTM metrics      [Python]
        ├──▶ valuation_agent      (:8001)   DCF · peers · analyst consensus    [Python]
        ├──▶ risk_agent           (:8003)   leverage · margins · red flags     [Python]
        └──▶ openrouter_research_agent (:8100)  general read     [TypeScript — Tier B]
   Python specialists = to_a2a services on OpenRouterLlm + FMP tools;
   the TS node = @openrouter/agent wrapped in @a2a-js/sdk (cross-runtime bridge).
```

## Pieces

| File | Role |
|------|------|
| `../finance_coordinator/models/openrouter_llm.py` | `OpenRouterLlm(BaseLlm)` — custom ADK model on the **OpenRouter client SDK** (`chat.send_async`), with tool-call mapping |
| `fundamentals_service.py` | fundamentals agent → A2A **server** (`:8002`) |
| `valuation_service.py` | valuation agent → A2A **server** (`:8001`) |
| `risk_service.py` | risk agent → A2A **server** (`:8003`) |
| `coordinator.py` | coordinator with a `RemoteA2aAgent` per specialist, routed by query |
| `run_demo.py` | boots all three services + runs the coordinator end-to-end, then tears down |

## Run

**One-shot demo** (boots the service for you):
```bash
uv run python a2a_finance/run_demo.py NVDA
uv run python a2a_finance/run_demo.py AAPL "is it cheap vs its DCF?"
```

**Or run the two processes yourself:**
```bash
# terminal 1 — the A2A service
uv run uvicorn a2a_finance.fundamentals_service:a2a_app --port 8002 &
uv run uvicorn a2a_finance.valuation_service:a2a_app   --port 8001 &
uv run uvicorn a2a_finance.risk_service:a2a_app        --port 8003 &
#   agent cards → http://localhost:800X/.well-known/agent-card.json

# terminal 2 — the coordinator (delegates over A2A, routed by query)
uv run python -c "import asyncio; from a2a_finance.coordinator import root_agent; ..."
```

## Adding future agents (the pattern)

1. Define an `LlmAgent` (any model — `OpenRouterLlm`, Gemini, LiteLLM).
2. Expose it: `app = to_a2a(agent, port=800X)` → serve with uvicorn.
3. Register it on the coordinator: `RemoteA2aAgent(name=…, agent_card="http://…/.well-known/agent-card.json")` in `sub_agents`.

Because agents talk over the protocol, a service can be a **different runtime** — done in **Tier B**: the TypeScript `OpenRouter-Agent` is exposed over A2A (via `@a2a-js/sdk`) and consumed here as `openrouter_research_agent`. See below.

## Tier B — cross-runtime bridge (Python ↔ TypeScript)

The `openrouter_research_agent` node is a **TypeScript** agent (`@openrouter/agent`) wrapped in an A2A server (`@a2a-js/sdk`), reachable by the Python coordinator over the same protocol. Nothing in the coordinator knows or cares that it's a different language — that's the point of A2A.

```bash
# 1) start the TypeScript A2A service (from OpenRouter-Agent/)
cd ../OpenRouter-Agent && npm run a2a          # → http://localhost:8100

# 2) start the 3 Python services + run the coordinator (from Google-ADK-agents/)
uv run python a2a_finance/run_demo.py NVDA "give me a general read"
#   → coordinator routes the broad read to the TS agent over A2A
```

Override the TS card URL with `A2A_OPENROUTER_CARD`. Because `RemoteA2aAgent` resolves cards lazily, the coordinator still runs with only the Python specialists if the TS service is down — it just can't route there.

## Config

| Env | Default | Purpose |
|-----|---------|---------|
| `OPENROUTER_API_KEY` | — | required (client SDK auth) |
| `FMP_API_KEY` | — | market-data tools |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct` | model for the coordinator |
| `A2A_{FUNDAMENTALS,VALUATION,RISK}_MODEL` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | model per service |
| `A2A_FUNDAMENTALS_PORT` / `A2A_VALUATION_PORT` / `A2A_RISK_PORT` | `8002` / `8001` / `8003` | service ports |

> **Model latency note:** the services default to the **Nemotron reasoning** model on OpenRouter's **free tier**, which queues (often several minutes) before responding — the reasoning + tool loop can take a while end-to-end. For a fast demo, override to a quicker model, e.g. `A2A_VALUATION_MODEL=meta-llama/llama-3.3-70b-instruct` (same for `A2A_FUNDAMENTALS_MODEL` / `A2A_RISK_MODEL`).

> A2A support in ADK is marked **experimental** (the warnings on startup are expected). Requires `a2a-sdk>=0.3.4,<0.4` and `sse-starlette` (both pinned in `pyproject.toml`).
