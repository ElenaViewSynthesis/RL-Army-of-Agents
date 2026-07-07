/**
 * OpenRouter finance agent — CLI entry point.
 *
 * Usage:
 *   npm run agent -- NVDA                      # streaming brief (default)
 *   npm run agent -- AAPL "is it expensive?"   # streaming, specific question
 *   npm run agent -- NVDA --structured          # zod-validated research note (JSON)
 *
 * `callModel` runs the full agentic loop: it sends the prompt, and whenever the
 * model calls one of our FMP tools it executes it, feeds the JSON back, and
 * continues. In the default mode we stream the answer to stdout as it is
 * generated. In `--structured` mode we add a `submit_research_note` tool whose
 * zod schema *is* the output contract, stop on that call, and print the
 * validated note — while the tool-gathering turns still stream to stderr.
 */

import { callModel, stepCountIs, hasToolCall, OpenRouter } from "@openrouter/agent";

import { financeTools } from "./tools.js";
import { researchNoteSchema, submitResearchNote } from "./schema.js";

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

const SYSTEM_STRUCTURED = `${SYSTEM}

After you have gathered the data you need, call the submit_research_note tool
exactly once with the completed note — that call is your final answer. Fill
every field; use null for priceTarget only when the data is genuinely
insufficient, and rank the risks most-severe first.`;

function parseArgs(argv: string[]) {
  const structured = argv.includes("--structured") || argv.includes("--json");
  const positional = argv.filter((a) => !a.startsWith("--"));
  const [ticker, ...rest] = positional;
  return { ticker, question: rest.join(" ").trim(), structured };
}

async function runStreaming(client: OpenRouter, ticker: string, prompt: string) {
  const result = callModel(client, {
    model: MODEL,
    instructions: SYSTEM,
    input: prompt,
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    tools: financeTools,
    stopWhen: stepCountIs(12),
    allowFinalResponse: true,
  });

  const toolLog = (async () => {
    for await (const tc of result.getToolCallsStream()) {
      console.error(`  [tool] ${tc.name}(${JSON.stringify(tc.arguments)})`);
    }
  })();

  console.error(`\n▸ ${MODEL} — researching ${ticker}…\n`);
  for await (const delta of result.getTextStream()) {
    process.stdout.write(delta);
  }
  process.stdout.write("\n");
  await toolLog;

  const u = (await result.getResponse()).usage;
  if (u) console.error(`\n▸ tokens — in: ${u.inputTokens ?? "?"}, out: ${u.outputTokens ?? "?"}`);
}

async function runStructured(client: OpenRouter, ticker: string, prompt: string) {
  const result = callModel(client, {
    model: MODEL,
    instructions: SYSTEM_STRUCTURED,
    input: prompt,
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    tools: [...financeTools, submitResearchNote],
    // Stop as soon as the note is submitted; bound the tool-gathering phase too.
    stopWhen: [hasToolCall("submit_research_note"), stepCountIs(12)],
    allowFinalResponse: false,
  });

  // Stream the model's intermediate reasoning/tool activity to stderr so the
  // run is observable; stdout is reserved for the final structured JSON.
  console.error(`▸ ${MODEL} — researching ${ticker} (structured)…\n`);
  const toolLog = (async () => {
    for await (const tc of result.getToolCallsStream()) {
      console.error(`  [tool] ${tc.name}`);
    }
  })();
  for await (const delta of result.getTextStream()) process.stderr.write(delta);
  await toolLog;

  const submit = (await result.getToolCalls()).find((c) => c.name === "submit_research_note");
  if (!submit) {
    console.error("\nModel did not submit a structured note.");
    process.exit(2);
  }
  // Validate the model's arguments against the schema before emitting.
  const note = researchNoteSchema.parse(submit.arguments);
  process.stdout.write("\n" + JSON.stringify(note, null, 2) + "\n");
}

async function main() {
  if (!process.env.OPENROUTER_API_KEY) {
    console.error("Error: OPENROUTER_API_KEY not set. Add it to OpenRouter-Agent/.env");
    process.exit(1);
  }

  const { ticker, question, structured } = parseArgs(process.argv.slice(2));
  if (!ticker) {
    console.error('Usage: npm run agent -- <TICKER> ["question"] [--structured]');
    process.exit(1);
  }
  const T = ticker.toUpperCase();
  const prompt = question
    ? `For ${T}: ${question}`
    : `Give me a research brief on ${T} — fundamentals, valuation, and the key risks, ending with a BUY/HOLD/SELL view.`;

  const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
  if (structured) await runStructured(client, T, prompt);
  else await runStreaming(client, T, prompt);
}

main().catch((e) => {
  console.error("\nAgent failed:", (e as Error).message);
  process.exit(1);
});
