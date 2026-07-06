/**
 * FMP-backed tools for the OpenRouter agent.
 *
 * Each `tool({...})` pairs a zod `inputSchema` with an async `execute`. The SDK
 * inspects these, exposes them to the model as callable functions, runs the
 * matching `execute` when the model calls one, and feeds the result back — all
 * inside a single `callModel` loop.
 */

import { tool } from "@openrouter/agent";
import { z } from "zod";

import { fmpGet, firstRecord } from "./fmp.js";

const symbolInput = z.object({
  symbol: z.string().describe('Stock ticker, e.g. "NVDA"'),
});

export const getCompanyProfile = tool({
  name: "get_company_profile",
  description: "Company profile: name, sector, industry, market cap, description.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) =>
    firstRecord(await fmpGet("profile", { symbol: symbol.toUpperCase() })),
});

export const getStockQuote = tool({
  name: "get_stock_quote",
  description: "Real-time quote: price, change, day range, volume.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) =>
    firstRecord(await fmpGet("quote", { symbol: symbol.toUpperCase() })),
});

export const getKeyMetrics = tool({
  name: "get_key_metrics",
  description: "TTM key metrics: P/E, ROE, margins, debt/equity, FCF yield.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) =>
    firstRecord(await fmpGet("key-metrics-ttm", { symbol: symbol.toUpperCase() })),
});

export const getDcfValuation = tool({
  name: "get_dcf_valuation",
  description: "Discounted-cash-flow fair value vs current price.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) =>
    firstRecord(await fmpGet("discounted-cash-flow", { symbol: symbol.toUpperCase() })),
});

export const getAnalystRatings = tool({
  name: "get_analyst_ratings",
  description: "Recent analyst grade actions and the consensus price target.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) => {
    const sym = symbol.toUpperCase();
    const [grades, consensus] = await Promise.all([
      fmpGet("grades", { symbol: sym, limit: 10 }),
      fmpGet("price-target-consensus", { symbol: sym }),
    ]);
    // FMP's /grades ignores `limit` and returns full history (1000s of rows),
    // which overflows smaller model context windows — bound it client-side.
    const recentGrades = Array.isArray(grades) ? grades.slice(0, 10) : grades;
    return { symbol: sym, recent_grades: recentGrades, price_target_consensus: firstRecord(consensus) };
  },
});

export const getPeers = tool({
  name: "get_peers",
  description: "List of peer tickers for competitive comparison.",
  inputSchema: symbolInput,
  execute: async ({ symbol }) => {
    const data = await fmpGet("stock-peers", { symbol: symbol.toUpperCase() });
    return Array.isArray(data) ? { symbol: symbol.toUpperCase(), peers: data } : data;
  },
});

export const financeTools = [
  getCompanyProfile,
  getStockQuote,
  getKeyMetrics,
  getDcfValuation,
  getAnalystRatings,
  getPeers,
] as const;
