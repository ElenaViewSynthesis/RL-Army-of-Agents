# Commodities Agent (OilPrice API)

A commodity-prices agent on ADK, backed by the live **[OilPrice API](https://oilpriceapi.com)** (460+ commodity codes across oil, gas, metals, coal, refined products, macro indicators, and more). Runs on the OpenRouter client SDK (`OpenRouterLlm`) — no Gemini needed.

## Tools (`tools.py`)

| Tool | OilPrice endpoint | Returns |
|------|-------------------|---------|
| `search_commodities(query)` | `/v1/commodities` (filtered) | codes matching a name/code substring |
| `list_commodities(category)` | `/v1/commodities` (filtered) | codes in a category (oil, gas, metal, …) |
| `get_commodity_price(code)` | `/v1/prices/latest` | latest price, unit, currency, timestamp |
| `get_commodity_history(code, period)` | `/v1/prices/{past_day\|past_week\|past_month\|past_year}` | historical price series |

Catalog/search output is capped (the full catalog is 460+ rows) and every tool returns `{"error": ...}` on a missing key / auth / network failure rather than raising.

## Setup

```bash
# add to finance_coordinator/.env (gitignored):
OILPRICE_API_KEY=your_oilprice_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Run

```bash
# standalone (from Google-ADK-agents/)
uv run python -m commodities_agent.run "price of Brent crude?"
uv run python -m commodities_agent.run "how has natural gas moved this month?"
uv run python -m commodities_agent.run "list metal commodities"

# or in the ADK dev UI (agent appears as commodities_agent)
uv run adk web
```

The agent searches for a commodity's code when needed, then fetches the latest price and/or history, and answers with the unit and currency (e.g. `$75.22 / barrel`).
