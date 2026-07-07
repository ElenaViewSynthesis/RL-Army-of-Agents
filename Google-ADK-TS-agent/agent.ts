/**
 * Finance chat agent on Google's TypeScript ADK (@google/adk).
 *
 * Main functionality is a **streaming chat** — run `npx adk web` and talk to it
 * in the browser UI (token-by-token streaming), or `npx adk run agent.ts` for a
 * streaming CLI chat. ADK's devtools provide the chat loop, so there is no
 * custom CLI here by design.
 *
 * `adk web` / `adk run` discover the exported `rootAgent`.
 */

import { FunctionTool, LlmAgent } from '@google/adk';
import { z } from 'zod';

const FMP = 'https://financialmodelingprep.com/stable';

/** Minimal FMP `/stable` GET — returns the first record or an `{ error }` object. */
async function fmpGet(path: string, symbol: string): Promise<unknown> {
  const key = process.env.FMP_API_KEY;
  if (!key) return { error: 'FMP_API_KEY not set — add it to .env' };
  try {
    const r = await fetch(`${FMP}/${path}?symbol=${symbol.toUpperCase()}&apikey=${key}`);
    if (r.status === 402) return { error: 'premium FMP endpoint' };
    if (!r.ok) return { error: `FMP HTTP ${r.status}` };
    const data = await r.json();
    return Array.isArray(data) ? (data[0] ?? { error: 'no data returned' }) : data;
  } catch (e) {
    return { error: `FMP request failed: ${(e as Error).message}` };
  }
}

const getStockQuote = new FunctionTool({
  name: 'get_stock_quote',
  description: 'Real-time quote: price, change, day range, volume for a ticker.',
  parameters: z.object({
    symbol: z.string().describe('Stock ticker, e.g. NVDA'),
  }),
  execute: ({ symbol }) => fmpGet('quote', symbol),
});

const getCompanyProfile = new FunctionTool({
  name: 'get_company_profile',
  description: 'Company profile: name, sector, industry, market cap, description.',
  parameters: z.object({
    symbol: z.string().describe('Stock ticker, e.g. NVDA'),
  }),
  execute: ({ symbol }) => fmpGet('profile', symbol),
});

export const rootAgent = new LlmAgent({
  name: 'finance_chat_agent',
  // Gemini model per the ADK quickstart. See README for the API-key / quota
  // caveat — this path needs a Gemini key with quota (GEMINI_API_KEY).
  model: 'gemini-flash-latest',
  description: 'Answers questions about public stocks using live market data.',
  instruction: `You are an equity research assistant in a live chat.
Use get_stock_quote and get_company_profile to fetch current data before
answering questions about a company. Be concise and specific, cite the numbers
you pulled, and if a tool reports missing data say so rather than inventing it.`,
  tools: [getStockQuote, getCompanyProfile],
});
