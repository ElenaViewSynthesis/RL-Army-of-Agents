# A2A Finance — inter-agent coordination over the A2A protocol

Tier-A [Agent2Agent (A2A)](https://google.github.io/A2A/) coordination for the finance agents. Specialist agents run as **independent A2A services**; a coordinator delegates to them **over HTTP** via `RemoteA2aAgent` — not in-process `sub_agents`. Everything runs on the **OpenRouter client SDK** (`OpenRouterLlm`), so no Gemini is needed.

```
finance_coordinator (OpenRouterLlm)              ── coordinator process ──
        │  transfer_to_agent  ──▶  A2A / HTTP
        ├──▶ fundamentals_agent   (:8002)   profile · quote · TTM metrics
        ├──▶ valuation_agent      (:8001)   DCF · peers · analyst consensus
        └──▶ risk_agent           (:8003)   leverage · margins · red flags
   each specialist = an independent A2A service (to_a2a), OpenRouterLlm + FMP tools
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

Because agents talk over the protocol, a future service could even be a **different runtime** (e.g. the TypeScript `OpenRouter-Agent`) as long as it speaks A2A — that's the cross-runtime bridge (Tier B).

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
