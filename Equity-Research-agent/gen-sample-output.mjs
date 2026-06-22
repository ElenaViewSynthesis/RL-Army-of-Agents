#!/usr/bin/env node
// One-shot: call W&I agent on OpenRouter, stream response, write to sample-outputs file.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// --- load .env manually ---
const envPath = path.join(__dirname, '.env');
for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
  const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)=(.*)$/);
  if (m && !m[1].startsWith('#')) process.env[m[1]] = m[2].trim();
}

const AGENT_FILE = path.join(__dirname, 'agents', 'transactional-liability-wi-agent.md');
const OUT_FILE   = path.join(__dirname, 'sample-outputs', 'wi-fintech-ai-startup-2026-06-22.md');
const MODEL      = 'nvidia/nemotron-3-ultra-550b-a55b:free';

const QUERY = `Price a W&I policy for a fintech AI startup acquisition at £80M enterprise value. The target processes personal financial data under GDPR. Provide a ROL indication, recommended retention, and key exclusions given the regulatory exposure.`;

const HEADER = `# W&I Policy — Fintech AI Startup Acquisition (£80M EV)

**Query:** ${QUERY}

**Agent:** Transactional Liability — Warranty & Indemnity Underwriter
**Model:** ${MODEL}
**Date:** 2026-06-22

---

`;

const systemPrompt = fs.readFileSync(AGENT_FILE, 'utf8');

console.error(`[gen-sample-output] Calling OpenRouter (${MODEL})...`);
console.error(`[gen-sample-output] Query: ${QUERY.slice(0, 80)}...`);

const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/ElenaViewSynthesis/RL-Army-of-Agents',
    'X-Title': 'Equity Research Agent',
  },
  body: JSON.stringify({
    model: MODEL,
    stream: true,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: QUERY },
    ],
  }),
  signal: AbortSignal.timeout(660_000),
});

if (!resp.ok) {
  const err = await resp.text();
  console.error(`[gen-sample-output] HTTP ${resp.status}: ${err}`);
  process.exit(1);
}

let content = '';
const decoder = new TextDecoder();

for await (const chunk of resp.body) {
  const text = decoder.decode(chunk, { stream: true });
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    if (!line.startsWith('data: ')) {
      console.error(`[raw] ${line}`);
      continue;
    }
    const data = line.slice(6);
    if (data === '[DONE]') { console.error('[gen] stream done'); break; }
    try {
      const parsed = JSON.parse(data);
      // log first chunk fully
      if (!content) console.error(`[first-chunk] ${JSON.stringify(parsed).slice(0, 300)}`);
      const delta = parsed.choices?.[0]?.delta?.content;
      if (delta) {
        content += delta;
        process.stderr.write('.');
      }
      const rt = parsed.usage?.reasoning_tokens ?? parsed.usage?.reasoningTokens;
      if (rt > 0) console.error(`\n  [Reasoning tokens: ${rt.toLocaleString()}]`);
    } catch (e) { console.error(`[parse-err] ${e.message} | ${data.slice(0, 100)}`); }
  }
}

console.error(`\n[gen-sample-output] Done. ${content.length} chars received.`);

if (!content.trim()) {
  console.error('[gen-sample-output] ERROR: empty response from model.');
  process.exit(1);
}

fs.writeFileSync(OUT_FILE, HEADER + content + '\n');
console.error(`[gen-sample-output] Saved → ${OUT_FILE}`);
console.log(content);
