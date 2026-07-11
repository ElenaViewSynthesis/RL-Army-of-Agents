# OilPrice API — provider & tools reference

Reference for the **commodities_agent** service. Convention: every third-party data provider added to this repo gets one of these MD files, co-located with its agent service — documenting the provider's API surface and the agent tools built on it.

- **Provider:** [OilPrice API](https://oilpriceapi.com) — real-time & historical commodity prices
- **Base URL:** `https://api.oilpriceapi.com/v1`
- **Auth:** header `Authorization: Token <OILPRICE_API_KEY>`
- **Key env var:** `OILPRICE_API_KEY` (in gitignored `finance_coordinator/.env`)
- **Client:** `commodities_agent/tools.py` (httpx)
- **Model:** `OpenRouterLlm` (OpenRouter client SDK) — no Gemini needed
- **Rate limit (free tier):** 10,000 requests/period; surfaced in `X-Ratelimit-{Limit,Remaining,Reset,Used,Tier}` response headers

---

## Catalog

**469 commodity codes across 22 categories.**

| Category | Count | | Category | Count |
|----------|------:|-|----------|------:|
| oil | 288 | | petrochemical | 18 |
| drilling_intelligence | 31 | | positioning | 14 |
| gas | 25 | | coal | 12 |
| macro_indicators | 23 | | futures | 11 |
| metal | 20 | | forex | 5 |
| marine_fuel | 4 | | refined_products | 3 |
| oil_product | 3 | | renewable_fuel | 2 |
| emissions | 2 | | fertilizer | 2 |
| natural_gas_intelligence | 1 | | oil_storage_intelligence | 1 |
| energy | 1 | | ngl | 1 |
| nuclear | 1 | | aviation_fuel | 1 |

**Common codes:** `BRENT_CRUDE_USD`, `BRENT_SPOT_USD`, `WTI_USD`, `NATURAL_GAS_USD`, `GOLD_USD`, `SILVER_USD`.

---

## API endpoints (verified)

### `GET /v1/commodities`
Full catalog of 460+ codes. **Envelope:** `{ "status": "success", "data": { "commodities": [ … ] } }`

Each item:
```json
{ "code": "BRENT_CRUDE_USD", "name": "Brent Crude Oil", "currency": "USD",
  "category": "oil", "description": "…", "unit": "barrel" }
```

### `GET /v1/prices/latest?by_code=<CODE>`
Latest price for one commodity. Without `by_code` it returns a default (Brent), so always pass `by_code`. **Envelope:** `{ "status": "success", "data": { … } }`
```json
{ "code": "BRENT_CRUDE_USD", "price": 75.22, "formatted": "$75.22",
  "currency": "USD", "unit": "barrel", "type": "spot_price",
  "created_at": "2026-07-10T22:21:34.200Z", "updated_at": "…" }
```

### `GET /v1/prices/{period}?by_code=<CODE>`
Historical series. `period` ∈ `past_day` · `past_week` · `past_month` · `past_year`. Returns up to ~100 points. **Envelope:** `{ "status": "success", "data": { "prices": [ … ] } }` where each point is `{ price, formatted, currency, code, unit, type, created_at, updated_at }`.

---

## Agent tools (`tools.py`)

| Tool | Endpoint | Signature → returns |
|------|----------|---------------------|
| `search_commodities` | `/v1/commodities` (filtered) | `(query: str)` → `{ query, count, truncated, matches[] }` — name/code substring match, capped at 30 |
| `list_commodities` | `/v1/commodities` (filtered) | `(category: str = "")` → `{ count, showing, truncated, commodities[] }` — by category, capped at 30 |
| `get_commodity_price` | `/v1/prices/latest` | `(code: str)` → `{ code, price, formatted, currency, unit, type, updated_at }` |
| `get_commodity_history` | `/v1/prices/{period}` | `(code: str, period: str = "past_week")` → `{ code, period, count, prices[] }` — `prices` slimmed to `{price, at}`, first 60 |

**Design notes**
- **Bounded output:** the catalog is 469 rows, so `search`/`list` cap at 30 and history slims to `{price, at}` × 60 — keeps tool results inside the model's context window.
- **Never raises:** on missing key, `401` auth, or network failure a tool returns `{"error": …}` so the agent reports the gap instead of the turn crashing.
- **Search-first:** the agent is instructed to call `search_commodities` to resolve a code before pricing when it isn't sure of the exact code.

---

## Provider-MD convention

When a new third-party provider is introduced for a new agent service:
1. Create `<service>/<PROVIDER>_API.md` (this format): provider overview, auth/env, verified endpoints with envelopes, and the agent tools mapping to them.
2. Note the key env var (kept in the gitignored `.env`), rate limits, and any output-bounding decisions.
3. Mark endpoints **verified** only after hitting them live.
