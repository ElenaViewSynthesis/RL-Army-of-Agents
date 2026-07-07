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
│   ├── agent.ts          # CLI: streaming (default) + --structured modes
│   ├── tools.ts          # 6 FMP tools via tool()
│   ├── schema.ts         # zod research-note schema + submit_research_note tool
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

**Streaming (default)** — the answer streams to stdout token by token:
```bash
npm run agent -- NVDA
npm run agent -- AAPL "is it expensive right now?"
```

**Structured output (zod)** — emit a schema-validated research note as JSON:
```bash
npm run agent -- NVDA --structured
```
A `submit_research_note` tool (its zod `inputSchema` *is* the output contract, `src/schema.ts`) is added to the loop; the model gathers data with the FMP tools, then calls it as its final answer. We stop on that call (`hasToolCall`), validate the arguments with `researchNoteSchema.parse`, and print the note. Fields: `ticker · rating · currentPrice · priceTarget · fundamentals · valuation · risks[] · confidence · summary`.

- Tool calls and token usage go to **stderr**, the answer/JSON to **stdout**, so you can split them:
  ```bash
  npm run agent -- NVDA 2>/dev/null > nvda-brief.md
  npm run agent -- NVDA --structured 2>/dev/null > nvda-note.json
  ```
- Model defaults to `~anthropic/claude-sonnet-latest`. Override with `OPENROUTER_MODEL` in `.env` (e.g. `nvidia/nemotron-3-ultra-550b-a55b`, `poolside/laguna-m.1`).

## Requirements

- Node.js 20.12+ (native `fetch`, `process.loadEnvFile`)
- OpenRouter API key — [openrouter.ai/keys](https://openrouter.ai/keys)
- FMP API key — [financialmodelingprep.com](https://financialmodelingprep.com) (free tier covers all 6 tools)
