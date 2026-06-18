import { OpenRouter } from '@openrouter/sdk';

const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
const FMP_KEY = process.env.FMP_API_KEY;

const ticker = (process.argv[2] || 'AAPL').toUpperCase();

const tool = {
  type: 'function',
  function: {
    name: 'get_stock_quote',
    description: 'Get real-time stock quote',
    parameters: {
      type: 'object',
      properties: { symbol: { type: 'string' } },
      required: ['symbol'],
    },
  },
};

console.log(`Testing OpenRouter (poolside/laguna-m.1:free) + FMP for ${ticker}\n`);

const t0 = Date.now();
const response = await client.chat.send({
  model: 'poolside/laguna-m.1:free',
  messages: [{ role: 'user', content: `Get the stock quote for ${ticker}.` }],
  tools: [tool],
  tool_choice: { type: 'function', function: { name: 'get_stock_quote' } },
});

const message = response.choices[0].message;
console.log(`OpenRouter response time: ${((Date.now() - t0) / 1000).toFixed(2)}s`);
console.log('finish_reason:', response.choices[0].finishReason);
console.log('tool_calls:', JSON.stringify(message.toolCalls, null, 2));

if (message.toolCalls?.length) {
  const tc = message.toolCalls[0];
  const { symbol } = JSON.parse(tc.function.arguments);
  const url = `https://financialmodelingprep.com/stable/quote?symbol=${symbol}&apikey=${FMP_KEY}`;
  const res = await fetch(url);
  const data = await res.json();
  console.log('\nFMP result:');
  console.log(JSON.stringify(data, null, 2));
} else {
  console.log('\nNo tool call returned. Model response:', message.content);
}

// ── Streaming test ────────────────────────────────────────────────────────────

const tickers = ['AAPL', 'NBIS'];

console.log('\n' + '─'.repeat(50));
console.log(`Streaming test — poolside/laguna-m.1:free`);
console.log(`Tickers: ${tickers.join(', ')}\n`);

const t1 = Date.now();
let ttft = null;

const stream = await client.chat.send({
  model: 'poolside/laguna-m.1:free',
  stream: true,
  messages: [
    {
      role: 'user',
      content: `Give me a brief investment overview of ${tickers.join(' and ')} stocks. For each: current sentiment, key risk, and one catalyst to watch.`,
    },
  ],
});

for await (const chunk of stream) {
  const content = chunk.choices?.[0]?.delta?.content;

  if (content) {
    if (!ttft) ttft = ((Date.now() - t1) / 1000).toFixed(2);
    process.stdout.write(content);
  }

  if (chunk.usage) {
    const elapsed = ((Date.now() - t1) / 1000).toFixed(2);
    const reasoning = chunk.usage.completionTokensDetails?.reasoningTokens ?? 0;
    console.log('\n' + '─'.repeat(50));
    console.log(`Time to first token : ${ttft}s`);
    console.log(`Total time          : ${elapsed}s`);
    console.log(`Prompt tokens       : ${chunk.usage.promptTokens}`);
    console.log(`Completion tokens   : ${chunk.usage.completionTokens}`);
    console.log(`Reasoning tokens    : ${reasoning}`);
  }
}
