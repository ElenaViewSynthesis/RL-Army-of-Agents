import { OpenRouter } from '@openrouter/sdk';
import * as weave from 'weave';
import { writeFileSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

function loadPrompt(promptPath) {
  const resolved = resolve(promptPath);
  try {
    return readFileSync(resolved, 'utf8').trim();
  } catch {
    throw new Error(`Prompt file not found: ${resolved}`);
  }
}

const FMP_KEY = process.env.FMP_API_KEY;
const STABLE = 'https://financialmodelingprep.com/stable';

// 429 and 5xx are transient — worth retrying.
// 4xx client errors (401, 403, 404) are permanent — fail immediately.
const FMP_RETRYABLE = new Set([429, 500, 503]);
const FMP_MAX_RETRIES = 3;
const FMP_BASE_DELAY  = 1000; // ms — doubles each attempt: 1s, 2s, 4s

async function fmpGet(url, params = {}) {
  const qs      = new URLSearchParams({ ...params, apikey: FMP_KEY }).toString();
  const sep     = url.includes('?') ? '&' : '?';
  const fullUrl = `${url}${sep}${qs}`;

  for (let attempt = 0; attempt <= FMP_MAX_RETRIES; attempt++) {
    let res;

    // Catch network-level failures (DNS, timeout, connection reset).
    // These are transient and get the same retry treatment as 503.
    try {
      res = await fetch(fullUrl);
    } catch (networkErr) {
      if (attempt === FMP_MAX_RETRIES) throw networkErr;
      const delay = FMP_BASE_DELAY * 2 ** attempt + Math.random() * FMP_BASE_DELAY;
      console.error(`  FMP network error (${networkErr.message}) — retry ${attempt + 1}/${FMP_MAX_RETRIES} in ${(delay / 1000).toFixed(1)}s`);
      await new Promise((r) => setTimeout(r, delay));
      continue;
    }

    // Success — parse and return immediately.
    if (res.ok) return res.json();

    // Permanent client error — no point retrying.
    if (!FMP_RETRYABLE.has(res.status) || attempt === FMP_MAX_RETRIES) {
      throw new Error(`FMP API ${res.status} ${res.statusText} — ${url}`);
    }

    // Transient error — compute how long to wait before next attempt.
    // Respect Retry-After header if the server sends one (common on 429).
    const retryAfter = res.headers.get('retry-after');
    const delay = retryAfter
      ? parseInt(retryAfter, 10) * 1000
      : FMP_BASE_DELAY * 2 ** attempt + Math.random() * FMP_BASE_DELAY;

    console.error(`  FMP ${res.status} — retry ${attempt + 1}/${FMP_MAX_RETRIES} in ${(delay / 1000).toFixed(1)}s`);
    await new Promise((r) => setTimeout(r, delay));
  }
}

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'get_market_indices',
      description:
        'Fetch real-time quotes for 9 global market symbols: ^VIX (CBOE Volatility Index), ' +
        '^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ Composite), ^RUT (Russell 2000), ' +
        '^FTSE (FTSE 100), ^STOXX50E (Euro STOXX 50), ^N225 (Nikkei 225), ^HSI (Hang Seng). ' +
        'Returns price, changePercentage, change, open, previousClose, dayLow, dayHigh, ' +
        'yearLow, yearHigh, priceAvg50, priceAvg200, volume, and timestamp for each symbol.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
];

async function executeTool(name) {
  if (name === 'get_market_indices') {
    const INDICES = ['^VIX', '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^STOXX50E', '^N225', '^HSI'];
    const results = await Promise.all(INDICES.map((s) => fmpGet(`${STABLE}/quote`, { symbol: s })));
    return results.flat();
  }
  throw new Error(`Unknown tool: ${name}`);
}

const DEFAULT_PROMPT_FILE = resolve(__dirname, 'prompts', 'indices-system-prompt.txt');

class IndicesModel extends weave.WeaveObject {
  constructor(systemPrompt) {
    super({
      name: 'global-indices-agent',
      description: 'Global market indices snapshot powered by Laguna via OpenRouter',
    });
    this.model = 'poolside/laguna-m.1:free';
    this.prompt = new weave.StringPrompt({
      name: 'indices-system-prompt',
      content: systemPrompt,
    });
    this._client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
    this.predict   = weave.op(this._predict.bind(this),   { name: 'predict'      });
    this.callTool  = weave.op(this._callTool.bind(this),  { name: 'indices_tool' });
  }

  async _predict(messages) {
    return this._client.chat.send({ model: this.model, messages, tools: TOOLS });
  }

  async _callTool(name) {
    return executeTool(name);
  }
}

async function runIndices(shouldSave, promptFile) {
  if (!FMP_KEY) throw new Error('FMP_API_KEY is not set. See .env.example.');
  if (!process.env.OPENROUTER_API_KEY) throw new Error('OPENROUTER_API_KEY is not set. See .env.example.');

  const systemPrompt = loadPrompt(promptFile);

  const weaveEnabled = !!process.env.WANDB_API_KEY;
  if (weaveEnabled) await weave.init('elenamylocuda-gemma/Financial MP');

  const model = new IndicesModel(systemPrompt);

  console.error(`\nGlobal Indices Agent`);
  console.error(`════════════════════════════════════`);
  console.error(`Model:  ${model.model} (OpenRouter)`);
  console.error(`Prompt: ${promptFile}`);
  console.error(`Weave:  ${weaveEnabled ? 'elenamylocuda-gemma/Financial MP ✓' : 'disabled (no WANDB_API_KEY)'}`);
  console.error(`════════════════════════════════════\n`);

  const messages = [
    { role: 'system', content: model.prompt.format() },
    {
      role: 'user',
      content:
        'Produce a complete global market indices snapshot report. ' +
        'Call get_market_indices to fetch live data, then write all 6 sections.',
    },
  ];

  let report = '';
  let iteration = 0;

  const MAX_ITERATIONS = 5;

  while (true) {
    iteration++;
    if (iteration > MAX_ITERATIONS) {
      throw new Error(
        `Agentic loop exceeded ${MAX_ITERATIONS} iterations without reaching a stop condition. ` +
        `Last finish_reason was 'tool_calls' — possible runaway tool-call cycle.`
      );
    }

    const response = await model.predict(messages);
    const message      = response.choices[0].message;
    const finishReason = response.choices[0].finishReason;
    messages.push(message);

    if (finishReason === 'stop' || finishReason === 'end_turn') {
      report = message.content || '';
      console.error(`\nReport complete.\n`);
      break;
    }

    if (finishReason === 'tool_calls') {
      const toolCalls = message.toolCalls || [];
      console.error(`[Step ${iteration}] Tool calls: ${toolCalls.length}`);

      const toolResults = await Promise.all(
        toolCalls.map(async (tc) => {
          try {
            const result  = await model.callTool(tc.function.name);
            const content = JSON.stringify(result);
            console.error(`  ✓ ${tc.function.name} — ${content.length} chars`);
            return { role: 'tool', toolCallId: tc.id, content };
          } catch (err) {
            console.error(`  ✗ ${tc.function.name} — ${err.message}`);
            return { role: 'tool', toolCallId: tc.id, content: `Error: ${err.message}` };
          }
        })
      );

      messages.push(...toolResults);
    }
  }

  console.log(report);

  if (shouldSave) {
    const date     = new Date().toISOString().slice(0, 10);
    const filename = `indices-agent-${date}.md`;
    writeFileSync(filename, report, 'utf8');
    console.error(`Saved: ${filename}`);
  }
}

const args       = process.argv.slice(2);
const shouldSave = args.includes('--save');
const promptArg  = args.find((a) => a.startsWith('--prompt='));
const promptFile = promptArg
  ? resolve(promptArg.split('=')[1])
  : DEFAULT_PROMPT_FILE;

runIndices(shouldSave, promptFile).catch((err) => {
  console.error(`\nError: ${err.message}`);
  console.error(`\nUsage: node indices-agent.js [--save] [--prompt=path/to/prompt.txt]`);
  process.exit(1);
});
