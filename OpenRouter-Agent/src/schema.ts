/**
 * Structured research-note schema (zod) + the tool that emits it.
 *
 * Structured output is done the zod-native way: `submit_research_note` is a tool
 * whose `inputSchema` *is* the research-note schema. We tell the model to call
 * it last, stop the loop on that call (`hasToolCall`), and read the validated
 * object from the tool arguments — so the final answer is guaranteed to match
 * the schema, while the tool-gathering turns still stream.
 */

import { tool } from "@openrouter/agent";
import { z } from "zod";

export const researchNoteSchema = z.object({
  ticker: z.string().describe("Ticker analyzed, e.g. NVDA"),
  rating: z.enum(["BUY", "HOLD", "SELL"]),
  currentPrice: z.number().nullable().describe("Latest price, or null if unavailable"),
  priceTarget: z
    .number()
    .nullable()
    .describe("12-month price target, or null if data is insufficient"),
  fundamentals: z
    .string()
    .describe("2-4 sentences: business, profitability, balance-sheet strength"),
  valuation: z
    .string()
    .describe("DCF fair value vs price, peer context, analyst consensus"),
  risks: z.array(z.string()).describe("Ranked list of the top material risks"),
  confidence: z.enum(["low", "medium", "high"]),
  summary: z.string().describe("Executive summary tying the sections together"),
});

export type ResearchNote = z.infer<typeof researchNoteSchema>;

export const submitResearchNote = tool({
  name: "submit_research_note",
  description:
    "Submit the completed research note. Call this exactly once, LAST, after " +
    "you have gathered data with the other tools. Its arguments are the final answer.",
  inputSchema: researchNoteSchema,
  // The value is irrelevant — we read the structured note from the call args.
  execute: async () => ({ received: true }),
});
