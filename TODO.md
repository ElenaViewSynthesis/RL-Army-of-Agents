# TODO

Ongoing build for the multi-agent stacks in this repo. Current state: **Tier-A A2A is done**
(`Google-ADK-agents/a2a_finance/`) — a coordinator delegates to fundamentals / valuation / risk
as independent A2A services, all on `OpenRouterLlm` (a custom ADK `BaseLlm` on the OpenRouter
client SDK).

## To implement (priority order)

1. ~~**Tier B — cross-runtime A2A bridge.**~~ ✅ Done — the TypeScript `OpenRouter-Agent`
   is exposed over A2A via `@a2a-js/sdk` (`src/a2a-server.ts`, `npm run a2a`, :8100) and
   registered on the Python coordinator as `openrouter_research_agent`. Verified end-to-end:
   Python `RemoteA2aAgent` → TS server → live FMP data. Used the official A2A JS SDK instead of
   hand-rolling the protocol.

2. **Broad "research TICKER" fan-out.** ⚠️ Mechanism done, needs speedup.
   `a2a_finance/research.py` — `research_agent` wraps each A2A specialist as an `AgentTool`
   (control returns after each, unlike `transfer_to_agent`) and synthesizes one note; run via
   `run_demo.py <TICKER> --research`. **Verified the fan-out calls specialists and control
   returns, but execution is sequential and times out** (didn't finish 3 A2A tool-loops + synth
   even at 9 min). **Follow-up: parallelize** — run the 3 specialists concurrently
   (`ParallelAgent`, sum→max latency) then synthesize, so it completes in a usable window.

3. **True streaming in `OpenRouterLlm`.** It yields a single `LlmResponse` per turn today
   (non-streaming). Add token streaming via the client SDK's streaming call, mapping partial
   chunks to partial `LlmResponse`s.
   → `Google-ADK-agents/finance_coordinator/models/openrouter_llm.py`

4. ~~**Reasoning handling in `OpenRouterLlm`.**~~ ✅ Done — requests OpenRouter `reasoning`
   and surfaces `msg.reasoning` as a genai *thought* part (`thought=True`), kept out of the
   answer text and not resent as content on later turns. Observe via `part.thought` on event
   parts. *(Still open: thread `reasoning_details` across turns for full multi-turn continuity.)*

5. **Premium FMP tools.** Port the SEC filings / ETF / mutual-fund / ownership endpoints (already
   implemented in `Equity-Research-agent/`) into the shared tool set.

## Blocked on external (not code)

- **Gemini prepay balance depleted** (`429 "prepayment credits are depleted"`). Blocks the
  in-process `finance_coordinator` Gemini agents and the `Google-ADK-TS-agent` chat. Fix = top up
  at <https://ai.studio/projects>. The A2A path is unaffected (runs on OpenRouter).

## Gotchas to remember

- A2A needs `a2a-sdk>=0.3.4,<0.4` (**not** 1.x — incompatible API) plus `sse-starlette`.
- ADK's A2A support is marked **experimental** (the startup warnings are expected).
- Structured output uses a **non-reasoning** model — a reasoning trace corrupts the JSON.
