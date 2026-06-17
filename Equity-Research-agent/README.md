# Equity Research Agent

An AI-powered equity research agent that produces institutional-grade research reports for any publicly traded stock. Powered by Claude Opus 4.8 and the Financial Modeling Prep (FMP) API.

## What it does

Given a ticker symbol, the agent:
1. Calls 13 FMP data endpoints in parallel (financials, valuation, insider trades, analyst ratings, news, peers, and more)
2. Runs an agentic loop where Claude autonomously decides what data to gather
3. Synthesizes all data into a comprehensive 10-section research report with a BUY/HOLD/SELL rating and 12-month price target

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
- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
- `FMP_API_KEY` — from [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs) (free tier: 250 req/day)

### About the FMP API

Financial Modeling Prep (FMP) is built for commercial use, serving developers, analysts, quants, fintech startups, investment banks and more. Access real-time and historical stock prices, company financials, market news, stock analysis tools and more. With over 100 endpoints, all the financial and stock market data you need is in one place.

**3. Load environment**
```bash
# On macOS/Linux
export $(cat .env | xargs)

# On Windows PowerShell
Get-Content .env | ForEach-Object { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) }
```

## Usage

```bash
# Print report to stdout
node agent.js AAPL

# Save report to a .md file (and print to stdout)
node agent.js NVDA --save

# Redirect report to file, progress to terminal
node agent.js MSFT --save > msft-report.md
```

**Note:** Node.js 18+ is required (uses native `fetch`).

## Example output

```
Equity Research Agent
════════════════════════════════════
Ticker: AAPL
Model:  claude-opus-4-8
════════════════════════════════════

[Step 1] Fetching data via 13 tool(s):
  → get_company_profile({"symbol":"AAPL"})
  → get_stock_quote({"symbol":"AAPL"})
  → get_income_statement({"symbol":"AAPL","period":"annual"})
  ...
  ✓ get_company_profile — 1842 chars
  ✓ get_stock_quote — 623 chars
  ...

Report generation complete.

# Apple Inc. (AAPL) — Equity Research
**Rating: BUY** | **12-Month Price Target: $245.00** | ...
```

## How it works

The agent uses the Anthropic SDK's tool use API in a manual agentic loop:

```
User: "Research AAPL"
  ↓
Claude → calls tools (parallel)
  ↓
Tool executor → fetches FMP data
  ↓
Claude → calls more tools if needed
  ↓
Claude → writes full research report
  ↓
Agent prints report
```

All tool calls in a single turn are executed in parallel with `Promise.all()`. The message history (including Claude's thinking blocks) is preserved across turns for full context continuity.

## FMP tools used

| Tool | Endpoint |
|------|----------|
| `get_company_profile` | `/api/v3/profile/{symbol}` |
| `get_stock_quote` | `/api/v3/quote/{symbol}` |
| `get_income_statement` | `/api/v3/income-statement/{symbol}` |
| `get_balance_sheet` | `/api/v3/balance-sheet-statement/{symbol}` |
| `get_cash_flow` | `/api/v3/cash-flow-statement/{symbol}` |
| `get_key_metrics` | `/api/v3/key-metrics-ttm/{symbol}` |
| `get_financial_ratios` | `/api/v3/ratios-ttm/{symbol}` |
| `get_dcf_valuation` | `/api/v3/discounted-cash-flow/{symbol}` |
| `get_analyst_ratings` | `/api/v3/grade/{symbol}` |
| `get_price_target` | `/api/v4/price-target-consensus` |
| `get_insider_trades` | `/api/v4/insider-trading` |
| `get_recent_news` | `/api/v3/stock_news` |
| `get_peers` | `/api/v4/stock_peers` |
