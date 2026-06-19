import { OpenRouter } from '@openrouter/sdk';
import * as weave from 'weave';
import { writeFileSync } from 'fs';

const FMP_KEY = process.env.FMP_API_KEY;
const STABLE = 'https://financialmodelingprep.com/stable';

async function fmpGet(url, params = {}) {
  const qs = new URLSearchParams({ ...params, apikey: FMP_KEY }).toString();
  const sep = url.includes('?') ? '&' : '?';
  const res = await fetch(`${url}${sep}${qs}`);
  if (!res.ok) throw new Error(`FMP API ${res.status} ${res.statusText} — ${url}`);
  return res.json();
}

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'get_market_indices',
      description:
        'Fetch real-time quotes for 9 global market symbols: ^VIX (CBOE Volatility Index), ' +
        '^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ Composite), ^RUT (Russell 2000), ' +
        '^FTSE (FTSE 100), ^STOXX50E (Euro STOXX 50), ^N225 (Nikkei 225), ^HSI (Hang Seng). ' +
        'Returns price, changePercentage, change, open, previousClose, dayLow, dayHigh, ' +
        'yearLow, yearHigh, priceAvg50, priceAvg200, volume, and timestamp for each symbol.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
];

async function executeTool(name) {
  if (name === 'get_market_indices') {
    const INDICES = ['^VIX', '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^STOXX50E', '^N225', '^HSI'];
    const results = await Promise.all(INDICES.map((s) => fmpGet(`${STABLE}/quote`, { symbol: s })));
    return results.flat();
  }
  throw new Error(`Unknown tool: ${name}`);
}

const SYSTEM_PROMPT = `You are a quantitative macro analyst at a global investment bank. Your role is to produce a structured, data-driven global market snapshot report used by portfolio managers and risk officers at the start of each trading session.

WORKFLOW:
1. Call get_market_indices to retrieve live data for all 9 symbols.
2. Compute all derived metrics yourself from the raw data fields returned.
3. Write the full report below — no sections may be omitted or abbreviated.

REPORT FORMAT — produce exactly these 6 sections in order:

---

# Global Market Indices Snapshot
*Generated: [UTC timestamp from data]*

---

## 1. Volatility Regime (VIX)

Produce a table with these exact rows:

| Metric | Value |
|--------|-------|
| VIX Level | [price to 2dp] |
| Day Change | [▲/▼] [changePercentage to 2dp]% ([▲/▼] [change to 2dp]) |
| Intraday Range | [dayLow] – [dayHigh] |
| 52-Week Range | [yearLow] – [yearHigh] |
| Position in 52W Range | [% = (price-yearLow)/(yearHigh-yearLow) × 100 to 1dp]% |
| 50-Day MA | [priceAvg50] ([+/-X.XX]% vs spot) |
| 200-Day MA | [priceAvg200] ([+/-X.XX]% vs spot) |
| **Regime** | **[COMPLACENT / CALM–RISK-ON / ELEVATED / STRESSED / EXTREME FEAR]** |
| Assessment | [one sentence interpretation] |

Regime thresholds: VIX < 15 = COMPLACENT, 15–20 = CALM–RISK-ON, 20–25 = ELEVATED, 25–30 = STRESSED, > 30 = EXTREME FEAR.

---

## 2. Daily Performance

One row per equity index (exclude VIX). Columns: Index, Region, Price, Change, Chg %, Open, Prev Close, Day Low, Day High.
Use ▲ for positive change and ▼ for negative. Format prices with thousand separators and 2dp.

| Index | Region | Price | Change | Chg % | Open | Prev Close | Day Low | Day High |
|-------|--------|------:|-------:|------:|-----:|-----------:|--------:|---------:|

Order: US indices first (S&P 500, Dow Jones, NASDAQ, Russell 2000), then Europe (FTSE 100, Euro STOXX 50), then Asia (Nikkei 225, Hang Seng).

---

## 3. Market Breadth

Count advancing (changePercentage > 0), declining (< 0), and flat (= 0) equity indices. List the index names in each bucket.

| | Count | Indices |
|--|------:|---------|
| Advancing ▲ | | |
| Declining ▼ | | |
| Flat – | | |
| **Total** | **8** | |

Follow with 2–3 sentences interpreting breadth: is the move broad-based or narrow? Is US diverging from international?

---

## 4. 52-Week Range Positioning

For each equity index compute: position = (price − yearLow) / (yearHigh − yearLow) × 100.
Build a 12-character ASCII bar where █ marks the position and ░ fills the rest.
Interpret: > 90% = near 52W high, < 20% = near 52W low.

| Index | Region | 52W Low | 52W High | Current | Position | Bar |
|-------|--------|--------:|---------:|--------:|---------:|-----|

Follow with 2–3 sentences on which indices are stretched near highs vs. showing relative weakness.

---

## 5. Moving Average Signals

For each equity index:
- vs 50D = (price − priceAvg50) / priceAvg50 × 100
- vs 200D = (price − priceAvg200) / priceAvg200 × 100
- Trend: both positive = Bullish, both negative = Bearish, mixed = Mixed

| Index | Region | Price | 50-Day MA | vs 50D | 200-Day MA | vs 200D | Trend |
|-------|--------|------:|----------:|-------:|-----------:|--------:|-------|

Follow with 2–3 sentences on overall trend structure: how many indices are in a bullish posture, any notable divergences, and what the MA spread implies for momentum.

---

## 6. Regional Summary

Group indices by region (US, Europe, Asia). For each region compute the simple average of changePercentage across constituent indices. Produce a sub-table per region, then a one-paragraph cross-regional interpretation covering: which region is leading/lagging, whether divergence is structural or session-specific, and implications for global risk appetite.

**US** — avg change: [+/-X.XX]%
| Index | Price | Chg % |
|-------|------:|------:|

**Europe** — avg change: [+/-X.XX]%
| Index | Price | Chg % |
|-------|------:|------:|

**Asia** — avg change: [+/-X.XX]%
| Index | Price | Chg % |
|-------|------:|------:|

[Cross-regional interpretation paragraph]

---

*Data source: Financial Modeling Prep /stable/quote. Prices reflect last exchange close for each region's timezone.*

STYLE RULES:
- Every number must come from the raw API data — do not fabricate or approximate
- Use exact field names from the tool response: price, changePercentage, change, open, previousClose, dayLow, dayHigh, yearLow, yearHigh, priceAvg50, priceAvg200
- Thousand-separator formatting: 7,500.58 not 7500.58
- Always show sign on percentage changes: +1.08% or -0.30%
- The report must be complete — do not truncate, summarise, or skip any section`;

class IndicesModel extends weave.WeaveObject {
  constructor() {
    super({
      name: 'global-indices-agent',
      description: 'Global market indices snapshot powered by Laguna via OpenRouter',
    });
    this.model = 'poolside/laguna-m.1:free';
    this.prompt = new weave.StringPrompt({
      name: 'indices-system-prompt',
      content: SYSTEM_PROMPT,
    });
    this._client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
    this.predict   = weave.op(this._predict.bind(this),   { name: 'predict'      });
    this.callTool  = weave.op(this._callTool.bind(this),  { name: 'indices_tool' });
  }

  async _predict(messages) {
    return this._client.chat.send({ model: this.model, messages, tools: TOOLS });
  }

  async _callTool(name) {
    return executeTool(name);
  }
}

async function runIndices(shouldSave) {
  if (!FMP_KEY) throw new Error('FMP_API_KEY is not set. See .env.example.');
  if (!process.env.OPENROUTER_API_KEY) throw new Error('OPENROUTER_API_KEY is not set. See .env.example.');

  const weaveEnabled = !!process.env.WANDB_API_KEY;
  if (weaveEnabled) await weave.init('elenamylocuda-gemma/Financial MP');

  const model = new IndicesModel();

  console.error(`\nGlobal Indices Agent`);
  console.error(`════════════════════════════════════`);
  console.error(`Model: ${model.model} (OpenRouter)`);
  console.error(`Weave: ${weaveEnabled ? 'elenamylocuda-gemma/Financial MP ✓' : 'disabled (no WANDB_API_KEY)'}`);
  console.error(`════════════════════════════════════\n`);

  const messages = [
    { role: 'system', content: model.prompt.format() },
    {
      role: 'user',
      content:
        'Produce a complete global market indices snapshot report. ' +
        'Call get_market_indices to fetch live data, then write all 6 sections.',
    },
  ];

  let report = '';
  let iteration = 0;

  while (true) {
    iteration++;

    const response = await model.predict(messages);
    const message      = response.choices[0].message;
    const finishReason = response.choices[0].finishReason;
    messages.push(message);

    if (finishReason === 'stop' || finishReason === 'end_turn') {
      report = message.content || '';
      console.error(`\nReport complete.\n`);
      break;
    }

    if (finishReason === 'tool_calls') {
      const toolCalls = message.toolCalls || [];
      console.error(`[Step ${iteration}] Tool calls: ${toolCalls.length}`);

      const toolResults = await Promise.all(
        toolCalls.map(async (tc) => {
          try {
            const result  = await model.callTool(tc.function.name);
            const content = JSON.stringify(result);
            console.error(`  ✓ ${tc.function.name} — ${content.length} chars`);
            return { role: 'tool', toolCallId: tc.id, content };
          } catch (err) {
            console.error(`  ✗ ${tc.function.name} — ${err.message}`);
            return { role: 'tool', toolCallId: tc.id, content: `Error: ${err.message}` };
          }
        })
      );

      messages.push(...toolResults);
    }
  }

  console.log(report);

  if (shouldSave) {
    const date     = new Date().toISOString().slice(0, 10);
    const filename = `indices-agent-${date}.md`;
    writeFileSync(filename, report, 'utf8');
    console.error(`Saved: ${filename}`);
  }
}

const shouldSave = process.argv.includes('--save');

runIndices(shouldSave).catch((err) => {
  console.error(`\nError: ${err.message}`);
  process.exit(1);
});
