# A2A Finance — inter-agent coordination over the A2A protocol

Tier-A [Agent2Agent (A2A)](https://google.github.io/A2A/) coordination for the finance agents. Specialist agents run as **independent A2A services**; a coordinator delegates to them **over HTTP** via `RemoteA2aAgent` — not in-process `sub_agents`. Everything runs on the **OpenRouter client SDK** (`OpenRouterLlm`), so no Gemini is needed.

```
finance_coordinator (OpenRouterLlm)              ── process A ──
        │  transfer_to_agent  ──▶  A2A / HTTP
        ▼
valuation_agent (OpenRouterLlm + FMP tools)      ── process B (:8001) ──
   served as an A2A service via to_a2a()
```

## Pieces

| File | Role |
|------|------|
| `../finance_coordinator/models/openrouter_llm.py` | `OpenRouterLlm(BaseLlm)` — custom ADK model on the **OpenRouter client SDK** (`chat.send_async`), with tool-call mapping |
| `valuation_service.py` | valuation agent → exposed as an A2A **server** with `to_a2a` (ASGI app `a2a_app`) |
| `coordinator.py` | coordinator with a `RemoteA2aAgent` pointing at the service's agent card |
| `run_demo.py` | boots the service + runs the coordinator end-to-end, then tears down |

## Run

**One-shot demo** (boots the service for you):
```bash
uv run python a2a_finance/run_demo.py NVDA
uv run python a2a_finance/run_demo.py AAPL "is it cheap vs its DCF?"
```

**Or run the two processes yourself:**
```bash
# terminal 1 — the A2A service
uv run uvicorn a2a_finance.valuation_service:a2a_app --port 8001
#   agent card → http://localhost:8001/.well-known/agent-card.json

# terminal 2 — the coordinator (delegates over A2A)
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
| `A2A_VALUATION_MODEL` | `meta-llama/llama-3.3-70b-instruct` | model for the valuation service |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct` | model for the coordinator |
| `A2A_VALUATION_PORT` | `8001` | service port |

> A2A support in ADK is marked **experimental** (the warnings on startup are expected). Requires `a2a-sdk>=0.3.4,<0.4` and `sse-starlette` (both pinned in `pyproject.toml`).
