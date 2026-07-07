# Google ADK — TypeScript Finance Chat Agent

A **streaming chat** agent built on Google's **TypeScript** Agent Development Kit (`@google/adk`). Ask it about a public company in the browser chat UI and it fetches live market data (via FMP tools) and answers, streaming token by token.

> This is the **TypeScript** ADK (`@google/adk`) — a different library from the Python ADK in [`../Google-ADK-agents`](../Google-ADK-agents) and from the OpenRouter SDK agent in [`../OpenRouter-Agent`](../OpenRouter-Agent).

## Main functionality: streaming chat

The chat loop is provided by ADK's devtools — there is **no custom CLI** here by design:

```bash
npx adk web            # browser chat UI at http://localhost:8000 (streaming)
npx adk run agent.ts   # streaming chat in the terminal
```

`adk web` serves a chat interface; pick `finance_chat_agent` in the top-right and start typing. Try: *"What's NVDA trading at?"*, *"Give me a quick read on AAPL."*

## Files

```
Google-ADK-TS-agent/
├── package.json      # ESM; main = agent.ts; scripts: chat (adk web), cli (adk run)
├── agent.ts          # rootAgent = LlmAgent + 2 FMP FunctionTools
└── .env.example
```

`agent.ts` exports `rootAgent` (an `LlmAgent`) with two `FunctionTool`s — `get_stock_quote` and `get_company_profile` — each a zod-typed FMP `/stable` call.

## Setup

```bash
cd Google-ADK-TS-agent
npm install @google/adk zod
npm install -D @google/adk-devtools

cp .env.example .env    # then fill in GEMINI_API_KEY and FMP_API_KEY
```

## Requirements

- **Node.js 24.13.0+** and **npm 11.8.0+** (per the ADK TS quickstart).
- `GEMINI_API_KEY` — Gemini key from [AI Studio](https://aistudio.google.com/app/apikey).
- `FMP_API_KEY` — [financialmodelingprep.com](https://financialmodelingprep.com).

## Prerequisites & known limitations

**1. API-key name — aligned.** `@google/adk` uses **`GEMINI_API_KEY`**. The sibling **Python** ADK project now uses the same name (its `google-genai` backend accepts `GEMINI_API_KEY`), so both projects share one env-var name — no mismatch to reconcile. Both run against AI Studio (`GOOGLE_GENAI_USE_VERTEXAI=FALSE`); the value can be the same key.

**2. Gemini billing balance.** This agent's model (`gemini-flash-latest`) needs a funded Gemini account. Billing is set up on the key used in this repo, but it currently returns `429 RESOURCE_EXHAUSTED — "prepayment credits are depleted"`, so the streaming chat authenticates but can't generate until the prepay balance is topped up at [AI Studio billing](https://ai.studio/projects). Same blocker as the Python coordinator's Gemini path; not a code issue. Unlike the Python project, the TS ADK quickstart is **Gemini-first**, so there is no OpenRouter/LiteLLM fallback wired here.

**3. Node version.** The quickstart requires **Node 24.13+**; installing on **Node 20** fails during a transitive `@google/genai` preinstall step on Windows (`STATUS_DLL_INIT_FAILED`). Install and run this under Node 24 (e.g. in WSL with `nvm use 24`).
