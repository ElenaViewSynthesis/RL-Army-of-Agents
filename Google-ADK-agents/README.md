# Google ADK Agents

Multi-agent financial-research orchestrator built on **Google's Agent Development Kit (ADK 2.3)**. A coordinator agent routes each request to the right specialist via ADK's LLM-driven delegation (`sub_agents`), then synthesizes the results.

Two selectable agents ship in this project:

## 1. `finance_coordinator` — interactive router

```
finance_coordinator          (root_agent — pure router, no data tools)
├── fundamentals_agent       profile · quote · TTM key metrics
├── valuation_agent          DCF fair value · peers · analyst consensus
└── risk_agent               leverage · margin fragility · red flags
```

The coordinator holds no market tools itself — it delegates to the specialist whose `description` best matches the query. Best for ad-hoc questions ("is TSLA expensive?", "risks in AAPL?").

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
# edit finance_coordinator/.env → set GOOGLE_API_KEY (Gemini) and FMP_API_KEY (market data)
```

## Run

```bash
uv run adk web                          # browser dev UI — pick either agent from the dropdown

uv run adk run finance_coordinator      # interactive router (CLI)
uv run adk run report_pipeline          # full-report pipeline (CLI)
```

- Router — try: `is TSLA expensive right now?`, `what are the risks in AAPL?`
- Pipeline — try: `NVDA` or `full report on MSFT` → runs all four steps → rated note.

## Next steps

- Add the premium FMP endpoints (SEC filings, ETF/mutual-fund holdings, ownership) as further tools.
- Wire the pipeline into a `Runner` + session service for programmatic/API use.
