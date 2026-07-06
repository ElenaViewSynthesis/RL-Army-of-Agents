/**
 * Financial Modeling Prep (FMP) `/stable` client.
 *
 * Mirrors the calling convention used across this repo (`?symbol=…&apikey=…`).
 * The key is read from `FMP_API_KEY`. Helpers return either the parsed JSON or
 * a `{ error }` object so tools can report gaps instead of throwing — the model
 * then reasons about missing data rather than the turn crashing.
 */

const FMP_STABLE = "https://financialmodelingprep.com/stable";

export type FmpResult = unknown[] | Record<string, unknown>;

export async function fmpGet(
  path: string,
  params: Record<string, string | number> = {},
): Promise<FmpResult> {
  const key = process.env.FMP_API_KEY;
  if (!key) return { error: "FMP_API_KEY not set — add it to .env" };

  const qs = new URLSearchParams({
    ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])),
    apikey: key,
  });

  try {
    const resp = await fetch(`${FMP_STABLE}/${path}?${qs}`, {
      signal: AbortSignal.timeout(20_000),
    });
    if (resp.status === 402) {
      return { error: "premium FMP endpoint — not available on the free tier", path };
    }
    if (!resp.ok) {
      return { error: `FMP HTTP ${resp.status}`, detail: (await resp.text()).slice(0, 200) };
    }
    return (await resp.json()) as FmpResult;
  } catch (e) {
    return { error: `FMP request failed: ${(e as Error).message}` };
  }
}

/** Normalize an FMP list response to its first record (or pass an error through). */
export function firstRecord(data: FmpResult): Record<string, unknown> {
  if (!Array.isArray(data)) return data; // error dict
  if (data.length === 0) return { error: "no data returned by FMP" };
  return data[0] as Record<string, unknown>;
}
