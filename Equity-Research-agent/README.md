# Equity Research Agent

An AI-powered equity research agent that produces institutional-grade research reports for any publicly traded stock. Powered by **nvidia/nemotron-3-ultra-550b-a55b** (default) via OpenRouter and the **Financial Modeling Prep (FMP) stable API**.

## What it does

Given a ticker symbol, the agent:
1. Calls 14 FMP data endpoints in parallel (financials, valuation, analyst ratings, peers, global indices + VIX, and more)
2. Runs an agentic loop where the model autonomously decides what data to gather
3. Synthesizes all data into a comprehensive 10-section research report with a BUY/HOLD/SELL rating and 12-month price target
4. Streams reasoning tokens internally — the model thinks before it writes

## Report sections

1. Executive Summary (rating + price target)
2. Company Overview
3. Financial Performance (4-year trend tables)
4. Balance Sheet & Capital Structure
5. Valuation Analysis (DCF + peer multiples table)
6. Growth Outlook
7. Competitive Position (peer comparison table)
8. Insider Activity & Analyst Sentiment
9. Risk Factors
10. Recent Developments

## Quick start

### Terminal
```bash
# 1. Install Node dependencies
npm install

# 2. Copy and fill in your API keys
cp .env.example .env

# 3. Run a research report
set -a && source .env && set +a
node agent.js NVDA --save --model=nemotron
```

Reports are saved to `output/NVDA-research-YYYY-MM-DD.md`. Progress logs stream to stderr; the report goes to stdout.

### Chat UI (WSL / Linux)
```bash
# Start the FastAPI server (kills any existing process on port 8000 first)
kill $(lsof -t -i:8000) 2>/dev/null; bash start.sh
```

Then open **http://localhost:8000/** in your browser. Type a question in the input field and press Enter — responses stream token by token. A live timer shows how long the model has been processing. Use the **Cancel** button to abort at any time.

To run a full equity research report via the API:
```bash
curl -s -X POST "http://localhost:8000/research/NVDA?model=nemotron&save=true"
```

---

## Setup

**1. Install dependencies**
```bash
npm install
```

**2. Set environment variables**
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

You need two API keys:
- `OPENROUTER_API_KEY` — from [openrouter.ai/keys](https://openrouter.ai/keys)
- `FMP_API_KEY` — from [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs)

## Usage

### Model selection

The default model is Nemotron (reasoning + streaming). Switch with `--model`:

```bash
node agent.js AAPL --model=nemotron   # default — reasoning, streaming
node agent.js AAPL --model=laguna     # faster, no reasoning tokens
node agent.js AAPL --model=nvidia/nemotron-3-ultra-550b-a55b:free  # full ID
```

### Single stock

```bash
# Using run.sh (recommended — auto-loads .env)
bash run.sh AAPL
bash run.sh NVDA --save        # saves report to output/NVDA-research-YYYY-MM-DD.md
bash run.sh MSFT --save        # saves and prints

# Or manually with env exported
set -a && source .env && set +a
node agent.js AAPL
node agent.js NVDA --save
```

### Multiple stocks — sequential

Run one after another, each saves its own `.md` file:

```bash
for ticker in AAPL NVDA MSFT GOOGL; do
  bash run.sh $ticker --save
done
```

### Multiple stocks — parallel

Run all at the same time in the background:

```bash
bash run.sh AAPL --save &
bash run.sh NVDA --save &
bash run.sh MSFT --save &
wait && echo "All reports done"
```

### Redirect output

Progress logs go to `stderr`, the report goes to `stdout` — so you can split them:

```bash
# Save report to file, watch progress in terminal
node agent.js AAPL --save 2>/dev/null > aapl-report.md

# Save both separately
node agent.js AAPL 2>progress.log > aapl-report.md
```

## FastAPI server

Serves the agent as a REST API with a built-in chat UI.

**Start (WSL / Linux)**
```bash
# First run or after port conflict
kill $(lsof -t -i:8000) 2>/dev/null; bash start.sh
```

`start.sh` installs Python deps, creates/repairs the venv, and starts uvicorn. On subsequent runs it auto-kills any existing process on port 8000.

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Chat UI — input field with streaming responses |
| `http://localhost:8000/docs` | Interactive API docs |
| `POST /research/{ticker}` | Full equity report as JSON |
| `GET /research/{ticker}/stream` | SSE stream — tool call progress + report |
| `POST /indices` | Global indices snapshot |
| `GET /output` | List saved reports |
| `GET /output/{filename}` | Fetch a saved report |
| `POST /chat/stream` | Direct chat completions (SSE) |

**Check OpenRouter credits**
```bash
node check-credits.js
```

## Test script

`test.js` validates both connections (OpenRouter + FMP) without running a full report:

```bash
set -a && source .env && set +a

# Tool call test (AAPL quote via Laguna) + streaming overview of AAPL and NBIS
node test.js

# Custom ticker for tool call section
node test.js TSLA
```

Output shows:
- OpenRouter response time
- `finish_reason` and tool call JSON
- Live FMP quote data
- Streamed investment overview for AAPL and NBIS with token usage + reasoning token count

## How it works

```
User: "Research AAPL"
  ↓
OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free — default)
  ↓ tool_calls (streaming, with reasoning tokens)
FMP /stable API — fetches 14 data points
  ↓ tool results
Model reasons + writes full report
  ↓
Agent prints / saves .md to output/
```

Tool calls within a single turn are executed in parallel with `Promise.all()`. The message history is preserved across turns. The model uses OpenAI-compatible function calling format (`type: 'function'`, `toolCalls`, `finishReason`).

## FMP tools — stable endpoints

All endpoints use `https://financialmodelingprep.com/stable` base URL with `?symbol=` query params.

| Tool | Endpoint | Available |
|------|----------|-----------|
| `get_company_profile` | `/stable/profile` | ✓ |
| `get_stock_quote` | `/stable/quote` | ✓ |
| `get_income_statement` | `/stable/income-statement` | ✓ |
| `get_balance_sheet` | `/stable/balance-sheet-statement` | ✓ |
| `get_cash_flow` | `/stable/cash-flow-statement` | ✓ |
| `get_key_metrics` | `/stable/key-metrics-ttm` | ✓ |
| `get_financial_ratios` | `/stable/ratios-ttm` | ✓ |
| `get_dcf_valuation` | `/stable/discounted-cash-flow` | ✓ |
| `get_analyst_ratings` | `/stable/grades` | ✓ |
| `get_price_target` | `/stable/price-target-consensus` | ✓ |
| `get_peers` | `/stable/stock-peers` | ✓ |
| `get_market_indices` | `/stable/quote` (×9 symbols) | ✓ |
| `get_insider_trades` | — | ✗ requires paid FMP plan |
| `get_recent_news` | — | ✗ requires paid FMP plan |

## Requirements

- Node.js 18+ (uses native `fetch`)
- Python 3.12+ with `python3.12-venv` (for FastAPI server)
- OpenRouter account — `nvidia/nemotron-3-ultra-550b-a55b:free` (default) or `poolside/laguna-m.1:free`
- FMP API key (free tier: 250 req/day — enough for ~17 full reports/day)

> **OpenRouter free tier note:** `:free` models require a deposited balance (≥ $10) to unlock 1,000 req/day and avoid queue delays. With $0 balance requests may hang; the server applies a 30s timeout and surfaces the error in the UI.
