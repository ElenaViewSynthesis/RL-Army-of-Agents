# Agent showcase — how to launch each agent & prompts to try

Launch commands and curated prompts for every agent, grouped by runtime. All the OpenRouter-backed agents run **without Gemini** (Gemini agents need a funded key). Keys live in the gitignored `finance_coordinator/.env` (`OPENROUTER_API_KEY`, `FMP_API_KEY`, `OILPRICE_API_KEY`).

---

## 🛢️ Commodities agent — OilPrice API (Python ADK)

**Launch (standalone, streaming):**
```bash
# from Google-ADK-agents/
uv run python -m commodities_agent.run "<prompt>"
```

**⭐ Daily glimpse — all endpoints at once (priority: gasoil, UK gas, petroleum):**
```bash
uv run python -m commodities_agent.run \
  "Show the Fuse Energy watchlist prices, then report the past-day movement for GASOIL_USD, NATURAL_GAS_GBP, and BRENT_CRUDE_USD."
```
→ one turn calls `list_fuse_watchlist` + `get_commodity_history ×3`. Sample output:
```
Fuse Energy watchlist:
  UK Natural Gas        116.90p / therm
  TTF Natural Gas Spot  €49.71 / MWh
  Brent Crude           $75.22 / barrel
  ICE Gasoil (Rotterdam)$1043.00 / tonne
  UK Carbon (UK ETS)    £55.49 / tCO₂
  EU Carbon (EU ETS)    €79.20 / tCO₂
  Newcastle Coal (API6) $128.60 / metric_ton
Past-day: GASOIL 67 pts · UK gas 12 pts · Brent (market closed)
```

**More prompts:**
```bash
uv run python -m commodities_agent.run "Show me the Fuse Energy watchlist prices."
uv run python -m commodities_agent.run "What is WTI crude trading at, and how has it moved this month?"
uv run python -m commodities_agent.run "Search commodities for 'diesel' and price the Rotterdam gasoil benchmark."
uv run python -m commodities_agent.run "List the metal commodities and give me the gold price."
```

> Tip: split streams — `... 2>/dev/null` shows just the answer; stderr shows the tool calls.

---

## 🔀 A2A multi-agent system — coordinator over the protocol (Python ADK)

Independent A2A services that a coordinator delegates to over HTTP. `run_demo.py` boots the services and runs the coordinator, then tears down.

**Routing mode** — one query → the matching specialist:
```bash
# from Google-ADK-agents/  (boots :8001/:8002/:8003/:8004)
uv run python a2a_finance/run_demo.py NVDA "is it cheap or expensive?"
uv run python a2a_finance/run_demo.py AAPL "what are the key risks?"
uv run python a2a_finance/run_demo.py MSFT "how is the business doing?"
uv run python a2a_finance/run_demo.py WTI  "current price of WTI crude oil?"      # → commodities_agent
uv run python a2a_finance/run_demo.py TSLA "give me a general read"                # → TS agent (start it first, below)
```

**Research fan-out** — consult all equity specialists, synthesize one note:
```bash
# fast models recommended (Nemotron:free queues); override valuation/risk:
A2A_VALUATION_MODEL=meta-llama/llama-3.3-70b-instruct \
A2A_RISK_MODEL=meta-llama/llama-3.3-70b-instruct \
  uv run python a2a_finance/run_demo.py NVDA --research
```

**Run a single A2A service manually** (then inspect its agent card):
```bash
uv run uvicorn a2a_finance.commodities_service:a2a_app --port 8004
curl http://localhost:8004/.well-known/agent-card.json
```

---

## 🟦 Cross-runtime node — TypeScript agent over A2A (Node)

The Python coordinator can delegate to a **TypeScript** agent (`@openrouter/agent` wrapped in `@a2a-js/sdk`). Start it, then route to it:
```bash
# terminal 1 — from OpenRouter-Agent/
npm run a2a                      # → http://localhost:8100

# terminal 2 — from Google-ADK-agents/
uv run python a2a_finance/run_demo.py MSFT "give me a general read"   # routes to the TS agent
```

**Standalone (Node) — streaming & structured:**
```bash
# from OpenRouter-Agent/
npm run agent -- NVDA                    # streaming brief
npm run agent -- AAPL "is it expensive?" # streaming, specific
npm run agent -- NVDA --structured        # zod-validated research note (JSON)
```

---

## 📊 Equity specialists — direct runners (Python ADK)

**Valuation reasoning agent** (streaming or structured note):
```bash
# from Google-ADK-agents/
uv run python run_valuation.py NVDA                    # stream tokens
uv run python run_valuation.py AAPL -q "cheap vs DCF?"  # specific question
uv run python run_valuation.py TSLA --structured         # Pydantic ResearchNote (JSON)
```

**ADK dev UI** (browser, pick an agent from the dropdown):
```bash
uv run adk web        # needs a funded Gemini key for the Gemini-backed agents
```

---

## Quick reference — ports & keys

| Service | Port | Key(s) |
|---------|------|--------|
| valuation / fundamentals / risk / commodities (A2A) | 8001 / 8002 / 8003 / 8004 | `OPENROUTER_API_KEY`, `FMP_API_KEY`, `OILPRICE_API_KEY` |
| TypeScript OpenRouter agent (A2A) | 8100 | `OPENROUTER_API_KEY`, `FMP_API_KEY` |

See each service's README / `*_API.md` for provider details.
