// import AnthropicBedrock from '@anthropic-ai/bedrock-sdk';
// import Anthropic from '@anthropic-ai/sdk';
import { OpenRouter } from '@openrouter/sdk';
import * as weave from 'weave';
import { writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { fmpGet, STABLE } from './lib/fmp.js';
import { runAgentLoop } from './lib/loop.js';
import { MODEL, WEAVE_PROJECT, INDICES } from './config.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = resolve(__dirname, 'output');
mkdirSync(OUTPUT_DIR, { recursive: true });

const FMP_KEY = process.env.FMP_API_KEY;

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'get_company_profile',
      description: 'Get company profile: sector, industry, description, CEO, employees, headquarters, website, and market cap',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string', description: 'Stock ticker symbol e.g. AAPL' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_stock_quote',
      description: 'Get real-time stock quote: current price, day range, 52-week range, volume, market cap, PE ratio',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_income_statement',
      description: 'Get income statements (last 4 periods): revenue, gross profit, operating income, EBITDA, net income, EPS',
      parameters: {
        type: 'object',
        properties: {
          symbol: { type: 'string' },
          period: { type: 'string', enum: ['annual', 'quarter'], description: 'annual or quarter, defaults to annual' },
        },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_balance_sheet',
      description: 'Get balance sheets (last 4 periods): cash, total assets, total debt, shareholders equity, working capital',
      parameters: {
        type: 'object',
        properties: {
          symbol: { type: 'string' },
          period: { type: 'string', enum: ['annual', 'quarter'] },
        },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_cash_flow',
      description: 'Get cash flow statements (last 4 periods): operating cash flow, capex, free cash flow, dividends, buybacks',
      parameters: {
        type: 'object',
        properties: {
          symbol: { type: 'string' },
          period: { type: 'string', enum: ['annual', 'quarter'] },
        },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_key_metrics',
      description: 'Get TTM key financial metrics: PE, PB, PS, EV/EBITDA, ROE, ROA, ROIC, debt/equity, FCF yield, dividend yield',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_financial_ratios',
      description: 'Get TTM financial ratios: gross/net/operating margins, current ratio, quick ratio, interest coverage, asset turnover',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_dcf_valuation',
      description: 'Get intrinsic value from discounted cash flow (DCF) model and compare to current stock price',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_analyst_ratings',
      description: 'Get recent analyst grade ratings (Strong Buy/Buy/Hold/Sell) from investment banks with dates and firms',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_price_target',
      description: 'Get Wall Street consensus price target: high, low, median, and average price targets across all analysts',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_insider_trades',
      description: 'Get recent insider transactions (purchases and sales) by company executives, directors, and 10%+ shareholders',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_recent_news',
      description: 'Get recent news articles and headlines about the company from financial media',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_peers',
      description: 'Get the list of peer and competitor companies in the same industry sector for comparison',
      parameters: {
        type: 'object',
        properties: { symbol: { type: 'string' } },
        required: ['symbol'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_market_indices',
      description: 'Get real-time quotes for 8 major global indices (S&P 500, Dow Jones, NASDAQ, Russell 2000, FTSE 100, Nikkei 225, Hang Seng, Euro STOXX 50) plus the CBOE Volatility Index (VIX). Use this to assess macro market conditions, global risk sentiment, and whether the current environment is risk-on or risk-off. VIX above 20 signals elevated fear; above 30 is extreme stress.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
];

async function executeTool(name, input) {
  const sym = (input.symbol || '').toUpperCase();
  const period = input.period || 'annual';

  switch (name) {
    case 'get_company_profile':
      return fmpGet(`${STABLE}/profile`, { symbol: sym });
    case 'get_stock_quote':
      return fmpGet(`${STABLE}/quote`, { symbol: sym });
    case 'get_income_statement':
      return fmpGet(`${STABLE}/income-statement`, { symbol: sym, limit: 4, period });
    case 'get_balance_sheet':
      return fmpGet(`${STABLE}/balance-sheet-statement`, { symbol: sym, limit: 4, period });
    case 'get_cash_flow':
      return fmpGet(`${STABLE}/cash-flow-statement`, { symbol: sym, limit: 4, period });
    case 'get_key_metrics':
      return fmpGet(`${STABLE}/key-metrics-ttm`, { symbol: sym });
    case 'get_financial_ratios':
      return fmpGet(`${STABLE}/ratios-ttm`, { symbol: sym });
    case 'get_dcf_valuation':
      return fmpGet(`${STABLE}/discounted-cash-flow`, { symbol: sym });
    case 'get_analyst_ratings':
      return fmpGet(`${STABLE}/grades`, { symbol: sym, limit: 10 });
    case 'get_price_target':
      return fmpGet(`${STABLE}/price-target-consensus`, { symbol: sym });
    case 'get_insider_trades':
      return [{ note: 'Insider trading data requires FMP paid plan — not available on current subscription.' }];
    case 'get_recent_news':
      return [{ note: 'Stock news requires FMP paid plan — not available on current subscription.' }];
    case 'get_peers':
      return fmpGet(`${STABLE}/stock-peers`, { symbol: sym });
    case 'get_market_indices': {
      const results = await Promise.all(INDICES.map((i) => fmpGet(`${STABLE}/quote`, { symbol: i.sym })));
      return results.flat();
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

const SYSTEM_PROMPT = `You are a senior equity research analyst at a premier investment bank (Goldman Sachs, Morgan Stanley caliber). Your mandate is to produce institutional-grade equity research reports used by portfolio managers to make investment decisions.

WORKFLOW:
1. Call ALL 14 available tools to gather comprehensive data on the company. Do not skip any tool.
2. Always call get_market_indices first (or in parallel with other tools) to establish the macro backdrop — VIX level, global index performance, and risk sentiment frame the entire report.
3. You may call multiple tools in parallel to gather data efficiently.
3. After all data is gathered, synthesize it into a complete research report.

REPORT FORMAT — Output must follow this exact structure:

# [COMPANY NAME] ([TICKER]) — Equity Research
**Date:** [Today's date]
**Rating: [BUY / HOLD / SELL]** | **12-Month Price Target: $[X.XX]** | **Current Price: $[X.XX]** | **Upside/Downside: [+/-X%]**

---

## 1. Executive Summary
[3 paragraphs: investment thesis, key catalysts supporting your view, primary risks, and why the stock is mispriced today. Be direct with your recommendation rationale.]

## 2. Company Overview
[Business model, revenue streams and their relative contribution, competitive moat analysis, management quality assessment, geographic and segment exposure. Include a brief description of what makes this business unique.]

## 3. Financial Performance
[Revenue growth trajectory (include YoY % changes), gross/operating/net margin evolution, EBITDA trends, EPS growth. Use a formatted table for 4-year historical data. Highlight inflection points or trend changes.]

| Metric | FY[N-3] | FY[N-2] | FY[N-1] | FY[N] |
|--------|---------|---------|---------|-------|
| Revenue ($B) | | | | |
| Gross Margin | | | | |
| Operating Income ($B) | | | | |
| Net Income ($B) | | | | |
| EPS | | | | |
| FCF ($B) | | | | |

## 4. Balance Sheet & Capital Structure
[Cash position, total debt, net debt/EBITDA leverage ratio, debt maturity schedule if available, capital allocation priorities (dividends, buybacks, M&A), liquidity assessment. Assess financial health and capacity to invest.]

## 5. Valuation Analysis
[Current trading multiples vs. 3-year historical average and vs. peer group average (P/E, EV/EBITDA, P/FCF, P/S). DCF-derived intrinsic value. Derive your 12-month price target with methodology explained. Include a valuation table.]

| Metric | Current | 3-Yr Avg | Peer Avg | Premium/Discount |
|--------|---------|----------|----------|-----------------|
| P/E | | | | |
| EV/EBITDA | | | | |
| P/FCF | | | | |
| P/S | | | | |

## 6. Growth Outlook
[Total addressable market (TAM) size and penetration. Organic growth drivers — product pipeline, geographic expansion, pricing power. Key catalysts for the next 12 months. Management guidance vs. consensus expectations. Long-term earnings power estimate.]

## 7. Competitive Position
[Industry structure and competitive dynamics. This company's relative position. Peer comparison table on revenue growth, margins, and valuation. Key competitive threats. Durable advantages analysis (switching costs, network effects, cost advantages, intangibles).]

| Company | Ticker | Revenue Growth | EBITDA Margin | EV/EBITDA |
|---------|--------|----------------|---------------|-----------|
| [Subject] | | | | |
| [Peer 1] | | | | |
| [Peer 2] | | | | |

## 8. Insider Activity & Analyst Sentiment
[Insider transaction summary for past 90 days — note any large purchases or systematic selling patterns. Analyst rating distribution (Strong Buy / Buy / Hold / Sell counts). Recent rating changes and their significance. Wall Street consensus price target range.]

## 9. Risk Factors
[Enumerate 5-8 specific risks with severity (High/Medium/Low) and probability assessment. Include: macro/cyclical risks, competitive risks, regulatory/legal risks, execution risks, financial risks. For each risk, briefly note any mitigation.]

## 10. Recent Developments
[Last 30 days of significant events: earnings releases, product announcements, M&A activity, regulatory actions, key partnerships, management changes. Assess market impact and implications for the thesis.]

---

## Investment Summary

| | |
|---|---|
| Rating | [BUY / HOLD / SELL] |
| Price Target (12M) | $X.XX |
| Current Price | $X.XX |
| Upside/Downside | +/-X% |
| Market Cap | $XB |
| Key Catalyst | [One sentence] |
| Primary Risk | [One sentence] |

*This report is for informational purposes only and does not constitute investment advice or a solicitation to buy or sell securities.*

STYLE REQUIREMENTS:
- Use precise numbers from the data (cite actual figures, not vague descriptors)
- Format large numbers as $XB (billions) or $XM (millions) for readability
- Express percentages to one decimal place
- Use comparative language: "Revenue grew 23% YoY to $94.8B vs. peer median of 12%"
- Be opinionated — analysts who say "it depends" are useless. Take a clear stance.`;

// ── ResearchModel ─────────────────────────────────────────────────────────────
// JS equivalent of Python's class JsonModel(weave.Model):
//   prompt: weave.Prompt = weave.StringPrompt(...)
//   @weave.op def predict(...)
class ResearchModel extends weave.WeaveObject {
  constructor() {
    super({ name: 'equity-research-agent', description: 'Institutional equity research powered by Laguna via OpenRouter' });
    this.model = MODEL;
    // StringPrompt is versioned and stored in Weave — visible in the UI alongside traces
    this.prompt = new weave.StringPrompt({
      name: 'equity-research-system-prompt',
      content: SYSTEM_PROMPT,
    });
    this._client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
    // @weave.op equivalent — wraps the method so every call is a logged trace
    this.predict = weave.op(this._predict.bind(this), { name: 'predict' });
    this.callTool = weave.op(this._callTool.bind(this), { name: 'fmp_tool' });
  }

  async _predict(messages) {
    return this._client.chat.send({ model: this.model, messages, tools: TOOLS });
  }

  async _callTool(name, input) {
    return executeTool(name, input);
  }
}

async function runResearch(ticker, shouldSave) {
  const symbol = ticker.toUpperCase();

  if (!FMP_KEY) {
    throw new Error('FMP_API_KEY environment variable is not set. See .env.example for setup instructions.');
  }
  if (!process.env.OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY environment variable is not set. See .env.example for setup instructions.');
  }

  const weaveEnabled = !!process.env.WANDB_API_KEY;
  if (weaveEnabled) {
    await weave.init(WEAVE_PROJECT);
  }

  const model = new ResearchModel();

  console.error(`\nEquity Research Agent`);
  console.error(`════════════════════════════════════`);
  console.error(`Ticker: ${symbol}`);
  console.error(`Model:  ${model.model} (OpenRouter)`);
  console.error(`Weave:  ${weaveEnabled ? WEAVE_PROJECT + ' ✓' : 'disabled (no WANDB_API_KEY)'}`);
  console.error(`════════════════════════════════════\n`);

  const messages = [
    { role: 'system', content: model.prompt.format() },
    {
      role: 'user',
      content: `Conduct a comprehensive equity research analysis for the stock ticker ${symbol}. Use ALL available tools to gather complete financial data before writing the report.`,
    },
  ];

  const report = await runAgentLoop(model, messages, { maxIterations: 10 });

  console.log(report);

  if (shouldSave) {
    const date     = new Date().toISOString().slice(0, 10);
    const filename = resolve(OUTPUT_DIR, `${symbol}-research-${date}.md`);
    writeFileSync(filename, report, 'utf8');
    console.error(`\nSaved: ${filename}`);
  }
}

// CLI
const args = process.argv.slice(2);
const ticker = args.find((a) => !a.startsWith('--'));
const shouldSave = args.includes('--save');

if (!ticker) {
  console.error('Usage: node agent.js <TICKER> [--save]');
  console.error('');
  console.error('Examples:');
  console.error('  node agent.js AAPL');
  console.error('  node agent.js NVDA --save');
  console.error('  node agent.js MSFT --save > msft-report.md');
  process.exit(1);
}

runResearch(ticker, shouldSave).catch((err) => {
  console.error(`\nError: ${err.message}`);
  process.exit(1);
});
