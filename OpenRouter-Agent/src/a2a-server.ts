/**
 * A2A server exposing this OpenRouter agent as an A2A service (Tier B).
 *
 * Lets a **Python ADK coordinator** delegate to this **TypeScript** agent over
 * the A2A protocol — a genuine cross-runtime bridge. Built on `@a2a-js/sdk`
 * (the official A2A JS SDK), mirroring how Python's `to_a2a` works: an
 * `AgentExecutor` runs the existing `callModel` tool loop and publishes the
 * result as an A2A message.
 *
 * Run:
 *   npm run a2a          # → http://localhost:8100
 *   agent card:  http://localhost:8100/.well-known/agent-card.json
 */

import { randomUUID } from "node:crypto";

import express from "express";
import {
  type AgentExecutor,
  type RequestContext,
  type ExecutionEventBus,
  DefaultRequestHandler,
  InMemoryTaskStore,
} from "@a2a-js/sdk/server";
import { A2AExpressApp } from "@a2a-js/sdk/server/express";
import type { AgentCard, Message } from "@a2a-js/sdk";
import { callModel, stepCountIs, OpenRouter } from "@openrouter/agent";

import { financeTools } from "./tools.js";

try {
  process.loadEnvFile(new URL("../.env", import.meta.url));
} catch {
  /* rely on already-exported env vars */
}

const PORT = Number(process.env.A2A_OR_PORT ?? 8100);
const MODEL = process.env.OPENROUTER_MODEL ?? "meta-llama/llama-3.3-70b-instruct";
const MAX_OUTPUT_TOKENS = Number(process.env.OPENROUTER_MAX_TOKENS ?? 1500);

const SYSTEM = `You are an equity research analyst. Use the provided tools to
gather live data (profile, quote, key metrics, DCF, analyst ratings, peers)
before answering. Be concise and specific, cite the numbers you pulled, and if
a tool reports missing data say so rather than inventing figures.`;

const agentCard: AgentCard = {
  protocolVersion: "0.3.0",
  name: "openrouter_research_agent",
  description:
    "Cross-runtime (TypeScript) equity-research agent on the OpenRouter Agent " +
    "SDK: fetches live FMP data and answers questions about a public company.",
  url: `http://localhost:${PORT}`,
  version: "0.1.0",
  preferredTransport: "JSONRPC",
  capabilities: {},
  defaultInputModes: ["text/plain"],
  defaultOutputModes: ["text/plain"],
  skills: [
    {
      id: "equity_research",
      name: "Equity research",
      description:
        "Given a ticker or question, gathers profile/quote/metrics/DCF/peers/" +
        "analyst data via FMP tools and returns a concise analyst answer.",
      tags: ["finance", "equities", "research"],
      examples: ["Research NVDA", "Is AAPL expensive right now?"],
    },
  ],
};

/** Runs the existing callModel tool loop and publishes the result as an A2A message. */
class ResearchExecutor implements AgentExecutor {
  async execute(ctx: RequestContext, bus: ExecutionEventBus): Promise<void> {
    const prompt = (ctx.userMessage.parts ?? [])
      .filter((p): p is { kind: "text"; text: string } => p.kind === "text")
      .map((p) => p.text)
      .join(" ")
      .trim();

    const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
    const result = callModel(client, {
      model: MODEL,
      instructions: SYSTEM,
      input: prompt || "Provide a brief equity-research read.",
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      tools: financeTools,
      stopWhen: stepCountIs(12),
      allowFinalResponse: true,
    });

    const answer = await result.getText();

    const message: Message = {
      kind: "message",
      messageId: randomUUID(),
      role: "agent",
      parts: [{ kind: "text", text: answer }],
      contextId: ctx.contextId,
      taskId: ctx.taskId,
    };
    bus.publish(message);
    bus.finished();
  }

  async cancelTask(): Promise<void> {
    /* single-shot executor; nothing to cancel */
  }
}

function main() {
  if (!process.env.OPENROUTER_API_KEY) {
    console.error("Error: OPENROUTER_API_KEY not set. Add it to OpenRouter-Agent/.env");
    process.exit(1);
  }

  const handler = new DefaultRequestHandler(
    agentCard,
    new InMemoryTaskStore(),
    new ResearchExecutor(),
  );
  const app = express();
  new A2AExpressApp(handler).setupRoutes(app);
  app.listen(PORT, () => {
    console.error(`▸ A2A OpenRouter agent on http://localhost:${PORT}`);
    console.error(`  agent card → http://localhost:${PORT}/.well-known/agent-card.json`);
  });
}

main();
