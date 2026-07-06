# OpenRouter Finance Agent

A financial-research agent built on the **OpenRouter Agent SDK** (`@openrouter/agent`). One `callModel` call runs the full agentic loop — the SDK sends the prompt, executes any FMP tool the model calls, feeds the JSON back, and continues until the model produces a final answer.

## How it works

```
callModel(client, { model, instructions, input, tools })
        │
        ├─ model requests a tool → SDK runs execute() → result fed back
        │  (get_stock_quote, get_company_profile, get_key_metrics,
        │   get_dcf_valuation, get_analyst_ratings, get_peers)
        │
        └─ loop repeats until final text  → streamed to stdout
```

Tools are defined with `tool({ name, description, inputSchema, execute })` using zod schemas (`src/tools.ts`); each calls the live **FMP `/stable` API** (`src/fmp.ts`). The loop is bounded with `stopWhen: stepCountIs(12)` and `allowFinalResponse: true` so it always ends on a clean text turn.

## Layout

```
OpenRouter-Agent/
├── package.json          # type: module; "agent" script via tsx
├── tsconfig.json
├── src/
│   ├── agent.ts          # CLI: callModel loop, streams answer + logs tool calls
│   ├── tools.ts          # 6 FMP tools via tool()
│   └── fmp.ts            # FMP /stable client
└── .env.example
```

## Setup

```bash
cd OpenRouter-Agent
npm install
cp .env.example .env      # add OPENROUTER_API_KEY and FMP_API_KEY
```

## Run

```bash
npm run agent -- NVDA
npm run agent -- AAPL "is it expensive right now?"
```

- The answer streams to **stdout**; tool calls and token usage go to **stderr**, so you can split them:
  ```bash
  npm run agent -- NVDA 2>/dev/null > nvda-brief.md
  ```
- Model defaults to `~anthropic/claude-sonnet-latest`. Override with `OPENROUTER_MODEL` in `.env` (e.g. `nvidia/nemotron-3-ultra-550b-a55b`, `poolside/laguna-m.1`).

## Requirements

- Node.js 20.12+ (native `fetch`, `process.loadEnvFile`)
- OpenRouter API key — [openrouter.ai/keys](https://openrouter.ai/keys)
- FMP API key — [financialmodelingprep.com](https://financialmodelingprep.com) (free tier covers all 6 tools)
