# TODO

Ongoing build for the multi-agent stacks in this repo. Current state: **Tier-A A2A is done**
(`Google-ADK-agents/a2a_finance/`) — a coordinator delegates to fundamentals / valuation / risk
as independent A2A services, all on `OpenRouterLlm` (a custom ADK `BaseLlm` on the OpenRouter
client SDK).

## To implement (priority order)

1. **Tier B — cross-runtime A2A bridge.** Wrap the TypeScript `OpenRouter-Agent/`
   (`@openrouter/agent`) in an A2A **server** (agent-card endpoint + message/task handling per
   the A2A spec — hand-rolled, since there's no native support) so the Python coordinator can
   delegate to it via `RemoteA2aAgent` over the wire. The real cross-language bridge.
   *Est. ~12–14 steps.*

2. **Broad "research TICKER" fan-out.** The coordinator currently routes one query → one
   specialist. Add a mode that consults all three A2A specialists and synthesizes one research
   note — likely via `AgentTool` (call sub-agents as tools so control returns) rather than
   `transfer_to_agent` (which hands off control).

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
