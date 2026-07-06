/**
 * OpenRouter finance agent — CLI entry point.
 *
 * Usage:
 *   npm run agent -- NVDA
 *   npm run agent -- AAPL "is it expensive right now?"
 *
 * The SDK's `callModel` runs the full agentic loop: it sends the prompt, and
 * whenever the model calls one of our FMP tools it executes it, feeds the JSON
 * back, and continues — until the model produces a final text answer (or the
 * `stopWhen` bound is hit). We stream that text to stdout and log tool calls to
 * stderr so the two streams can be redirected separately.
 */

import { callModel, stepCountIs, OpenRouter } from "@openrouter/agent";

import { financeTools } from "./tools.js";

// Load .env (Node 20.12+). Non-fatal if the file is absent.
try {
  process.loadEnvFile(new URL("../.env", import.meta.url));
} catch {
  /* rely on already-exported env vars */
}

const MODEL = process.env.OPENROUTER_MODEL ?? "~anthropic/claude-sonnet-latest";
// Cap per-request output. The SDK otherwise requests up to 65536 tokens, which
// low-balance OpenRouter accounts reject upfront. Override with OPENROUTER_MAX_TOKENS.
const MAX_OUTPUT_TOKENS = Number(process.env.OPENROUTER_MAX_TOKENS ?? 1500);

const SYSTEM = `You are an equity research analyst. When asked about a stock,
use the provided tools to gather live data (profile, quote, key metrics, DCF,
analyst ratings, peers) before answering. Be concise and specific, cite the
numbers you pulled, and if a tool reports missing data say so plainly rather
than inventing figures.`;

async function main() {
  if (!process.env.OPENROUTER_API_KEY) {
    console.error("Error: OPENROUTER_API_KEY not set. Add it to OpenRouter-Agent/.env");
    process.exit(1);
  }

  const [ticker, ...rest] = process.argv.slice(2);
  if (!ticker) {
    console.error('Usage: npm run agent -- <TICKER> ["question"]');
    process.exit(1);
  }
  const question = rest.join(" ").trim();
  const prompt = question
    ? `For ${ticker.toUpperCase()}: ${question}`
    : `Give me a concise research brief on ${ticker.toUpperCase()} — fundamentals, valuation, and the key risks, ending with a BUY/HOLD/SELL view.`;

  const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });

  const result = callModel(client, {
    model: MODEL,
    instructions: SYSTEM,
    input: prompt,
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    tools: financeTools,
    // Bound the loop so a misbehaving model can't spin forever; still allow a
    // clean final text turn if the bound is reached mid-tool-call.
    stopWhen: stepCountIs(12),
    allowFinalResponse: true,
  });

  // Log tool activity to stderr, concurrently with streaming the answer.
  const toolLog = (async () => {
    for await (const tc of result.getToolCallsStream()) {
      console.error(`  [tool] ${tc.name}(${JSON.stringify(tc.arguments)})`);
    }
  })();

  console.error(`\n▸ ${MODEL} — researching ${ticker.toUpperCase()}…\n`);
  for await (const delta of result.getTextStream()) {
    process.stdout.write(delta);
  }
  process.stdout.write("\n");
  await toolLog;

  const resp = await result.getResponse();
  const u = resp.usage;
  if (u) {
    console.error(
      `\n▸ tokens — in: ${u.inputTokens ?? "?"}, out: ${u.outputTokens ?? "?"}`,
    );
  }
}

main().catch((e) => {
  console.error("\nAgent failed:", (e as Error).message);
  process.exit(1);
});
