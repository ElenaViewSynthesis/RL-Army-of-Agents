export const STABLE = 'https://financialmodelingprep.com/stable';

const RETRYABLE  = new Set([429, 500, 503]);
const MAX_RETRIES = 3;
const BASE_DELAY  = 1000; // ms — doubles each attempt: 1s, 2s, 4s

export async function fmpGet(url, params = {}) {
  const apiKey  = process.env.FMP_API_KEY;
  const qs      = new URLSearchParams({ ...params, apikey: apiKey }).toString();
  const sep     = url.includes('?') ? '&' : '?';
  const fullUrl = `${url}${sep}${qs}`;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    let res;

    // Catch network-level failures (DNS, timeout, connection reset).
    try {
      res = await fetch(fullUrl);
    } catch (networkErr) {
      if (attempt === MAX_RETRIES) throw networkErr;
      const delay = BASE_DELAY * 2 ** attempt + Math.random() * BASE_DELAY;
      console.error(`  FMP network error (${networkErr.message}) — retry ${attempt + 1}/${MAX_RETRIES} in ${(delay / 1000).toFixed(1)}s`);
      await new Promise((r) => setTimeout(r, delay));
      continue;
    }

    if (res.ok) return res.json();

    // Permanent client error (401, 403, 404) — no point retrying.
    if (!RETRYABLE.has(res.status) || attempt === MAX_RETRIES) {
      throw new Error(`FMP API ${res.status} ${res.statusText} — ${url}`);
    }

    // Transient error — respect Retry-After if present, else exponential backoff + jitter.
    const retryAfter = res.headers.get('retry-after');
    const delay = retryAfter
      ? parseInt(retryAfter, 10) * 1000
      : BASE_DELAY * 2 ** attempt + Math.random() * BASE_DELAY;

    console.error(`  FMP ${res.status} — retry ${attempt + 1}/${MAX_RETRIES} in ${(delay / 1000).toFixed(1)}s`);
    await new Promise((r) => setTimeout(r, delay));
  }
}
