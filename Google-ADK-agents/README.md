# Google ADK Agents

Multi-agent financial-research orchestrator built on **Google's Agent Development Kit (ADK 2.3)**. A coordinator agent routes each request to the right specialist via ADK's LLM-driven delegation (`sub_agents`), then synthesizes the results.

## Architecture

```
finance_coordinator          (root_agent — pure router, no data tools)
├── fundamentals_agent       profile · quote · TTM key metrics
├── valuation_agent          DCF fair value · peers · analyst consensus
└── risk_agent               leverage · margin fragility · red flags
```

The coordinator holds no market tools itself — it delegates to the specialist whose `description` best matches the query, mirroring the domain of the sibling `Equity-Research-agent`. Market tools call the **live Financial Modeling Prep (FMP) `/stable` API** (6 core endpoints: profile, quote, key-metrics-ttm, discounted-cash-flow, grades + price-target-consensus, stock-peers). Tools return `{"error": ...}` on a missing key / premium gate / network failure so the model reports the gap rather than crashing.

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
└── .env.example
```

## Setup

```bash
# from Google-ADK-agents/
uv sync                          # install google-adk into .venv

# ADK reads the .env next to the agent package
cp .env.example finance_coordinator/.env
# edit finance_coordinator/.env → set GOOGLE_API_KEY (Gemini) and FMP_API_KEY (market data)
```

## Run

```bash
uv run adk web                   # browser dev UI at http://localhost:8000
uv run adk run finance_coordinator   # interactive CLI
```

Try: `research NVDA`, `is TSLA expensive right now?`, `what are the risks in AAPL?` — the coordinator transfers to the matching specialist.

## Next steps

- Add the premium FMP endpoints (SEC filings, ETF/mutual-fund holdings, ownership) as further tools.
- Add a `SequentialAgent` pipeline for a full report (fundamentals → valuation → risk → synthesis).
- Wire the coordinator into a `Runner` + session service for programmatic/API use.
