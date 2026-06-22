import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env manually (no dotenv dependency)
const envPath = resolve(__dirname, '.env');
for (const line of readFileSync(envPath, 'utf8').split('\n')) {
  const [key, ...rest] = line.trim().split('=');
  if (key && !key.startsWith('#') && rest.length) {
    process.env[key] = rest.join('=').replace(/^['"]|['"]$/g, '');
  }
}

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  console.error('OPENROUTER_API_KEY not found in .env');
  process.exit(1);
}

const res = await fetch('https://openrouter.ai/api/v1/key', {
  headers: { Authorization: `Bearer ${apiKey}` },
});

if (!res.ok) {
  console.error(`HTTP ${res.status} ${res.statusText}`);
  process.exit(1);
}

const { data } = await res.json();

console.log(`\nOpenRouter API Key Info`);
console.log(`═══════════════════════════════════`);
console.log(`Label          : ${data.label ?? '(none)'}`);
console.log(`Usage (USD)    : $${data.usage?.toFixed(4) ?? '0.0000'}`);
console.log(`Credit limit   : ${data.limit !== null ? '$' + data.limit?.toFixed(2) : 'unlimited'}`);
console.log(`Remaining      : ${data.limit !== null ? '$' + (data.limit - (data.usage ?? 0)).toFixed(4) : 'unlimited'}`);
console.log(`Rate limit     : ${data.rate_limit?.requests ?? '?'} req / ${data.rate_limit?.interval ?? '?'}`);
console.log(`Is free tier   : ${data.is_free_tier ?? 'unknown'}`);
console.log(`═══════════════════════════════════\n`);
