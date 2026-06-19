import { writeFileSync } from 'fs';

const FMP_KEY = process.env.FMP_API_KEY;
if (!FMP_KEY) throw new Error('FMP_API_KEY not set');

const STABLE = 'https://financialmodelingprep.com/stable';

const SYMBOLS = [
  { sym: '^VIX',      label: 'VIX',          region: 'US'     },
  { sym: '^GSPC',     label: 'S&P 500',       region: 'US'     },
  { sym: '^DJI',      label: 'Dow Jones',     region: 'US'     },
  { sym: '^IXIC',     label: 'NASDAQ Comp.',  region: 'US'     },
  { sym: '^RUT',      label: 'Russell 2000',  region: 'US'     },
  { sym: '^FTSE',     label: 'FTSE 100',      region: 'Europe' },
  { sym: '^STOXX50E', label: 'Euro STOXX 50', region: 'Europe' },
  { sym: '^N225',     label: 'Nikkei 225',    region: 'Asia'   },
  { sym: '^HSI',      label: 'Hang Seng',     region: 'Asia'   },
];

const quotes = await Promise.all(
  SYMBOLS.map(async ({ sym, label, region }) => {
    const res = await fetch(`${STABLE}/quote?symbol=${encodeURIComponent(sym)}&apikey=${FMP_KEY}`);
    const data = await res.json();
    return { ...data[0], label, region };
  })
);

const now = new Date();
const ts  = now.toUTCString();
const dateSlug = now.toISOString().slice(0, 10);

// ── helpers ──────────────────────────────────────────────────────────────────

const fmt  = (n, dec = 2) => n != null ? n.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) : 'n/a';
const pct  = (n)          => n != null ? `${n >= 0 ? '+' : ''}${n.toFixed(2)}%` : 'n/a';
const dir  = (n)          => n == null ? '' : n > 0 ? '▲' : n < 0 ? '▼' : '–';
const yPct = (price, low, high) => price != null && high > low ? (((price - low) / (high - low)) * 100).toFixed(1) + '%' : 'n/a';
const maPct = (price, ma)       => price != null && ma ? (((price - ma) / ma) * 100).toFixed(2) + '%' : 'n/a';
const pad  = (s, n, right = false) => right ? String(s).padStart(n) : String(s).padEnd(n);

// ── VIX regime ───────────────────────────────────────────────────────────────

const vix = quotes.find(q => q.symbol === '^VIX');
const vixPrice = vix?.price ?? 0;
let regime, regimeNote;
if      (vixPrice < 15) { regime = 'COMPLACENT';       regimeNote = 'Historically low fear — markets pricing near-zero near-term risk.'; }
else if (vixPrice < 20) { regime = 'CALM / RISK-ON';   regimeNote = 'Below 20 threshold — benign volatility environment.'; }
else if (vixPrice < 25) { regime = 'ELEVATED';         regimeNote = 'Above 20 — investors pricing in moderate uncertainty.'; }
else if (vixPrice < 30) { regime = 'STRESSED';         regimeNote = 'Approaching 30 — meaningful market stress.'; }
else                    { regime = 'EXTREME FEAR';     regimeNote = 'Above 30 — historical crisis/panic territory.'; }

// ── breadth: how many indices are positive ────────────────────────────────────

const equities = quotes.filter(q => q.symbol !== '^VIX');
const advancing = equities.filter(q => q.changePercentage > 0).length;
const declining  = equities.filter(q => q.changePercentage < 0).length;
const flat       = equities.length - advancing - declining;

// ── build markdown ────────────────────────────────────────────────────────────

const lines = [];
const h  = (text, lvl = 2) => lines.push('\n' + '#'.repeat(lvl) + ' ' + text);
const ln = (text = '')      => lines.push(text);
const hr = ()               => lines.push('\n---');

ln(`# Global Market Indices Snapshot`);
ln(`*Generated: ${ts}*`);
hr();

// 1. Regime summary
h('1. Volatility Regime (VIX)', 2);
ln(`| Metric | Value |`);
ln(`|--------|-------|`);
ln(`| VIX Level | **${fmt(vixPrice)}** |`);
ln(`| Day Change | ${dir(vix?.changePercentage)} ${pct(vix?.changePercentage)} (${dir(vix?.change)} ${fmt(vix?.change)}) |`);
ln(`| Intraday Range | ${fmt(vix?.dayLow)} – ${fmt(vix?.dayHigh)} |`);
ln(`| 52-Week Range | ${fmt(vix?.yearLow)} – ${fmt(vix?.yearHigh)} |`);
ln(`| 50-Day MA | ${fmt(vix?.priceAvg50)} (${maPct(vixPrice, vix?.priceAvg50)} vs spot) |`);
ln(`| 200-Day MA | ${fmt(vix?.priceAvg200)} (${maPct(vixPrice, vix?.priceAvg200)} vs spot) |`);
ln(`| **Regime** | **${regime}** |`);
ln(`| Assessment | ${regimeNote} |`);

// 2. Price & daily performance
h('2. Price & Daily Performance', 2);
ln(`| Index | Region | Price | Change | Chg % | Open | Prev Close | Day Low | Day High |`);
ln(`|-------|--------|------:|-------:|------:|-----:|-----------:|--------:|---------:|`);
for (const q of equities) {
  const arrow = dir(q.changePercentage);
  ln(`| **${q.label}** | ${q.region} | ${fmt(q.price)} | ${arrow} ${fmt(Math.abs(q.change))} | ${arrow} ${Math.abs(q.changePercentage ?? 0).toFixed(2)}% | ${fmt(q.open)} | ${fmt(q.previousClose)} | ${fmt(q.dayLow)} | ${fmt(q.dayHigh)} |`);
}

// 3. Market breadth
h('3. Market Breadth (Equities Only, Excl. VIX)', 2);
ln(`| | Count | Indices |`);
ln(`|--|------:|---------|`);
ln(`| Advancing ▲ | ${advancing} | ${equities.filter(q => q.changePercentage > 0).map(q => q.label).join(', ') || '—'} |`);
ln(`| Declining ▼ | ${declining} | ${equities.filter(q => q.changePercentage < 0).map(q => q.label).join(', ') || '—'} |`);
ln(`| Flat – | ${flat} | ${equities.filter(q => q.changePercentage === 0).map(q => q.label).join(', ') || '—'} |`);
ln(`| **Total** | **${equities.length}** | |`);

// 4. 52-week positioning
h('4. 52-Week Range Positioning', 2);
ln(`| Index | Region | 52W Low | 52W High | Current | Position in Range |`);
ln(`|-------|--------|--------:|---------:|--------:|:-----------------:|`);
for (const q of equities) {
  const pos = yPct(q.price, q.yearLow, q.yearHigh);
  const bar = buildBar(q.price, q.yearLow, q.yearHigh, 12);
  ln(`| **${q.label}** | ${q.region} | ${fmt(q.yearLow)} | ${fmt(q.yearHigh)} | ${fmt(q.price)} | ${pos} \`${bar}\` |`);
}

// 5. Moving average signals
h('5. Moving Average Signals', 2);
ln(`| Index | Region | Price | 50-Day MA | vs 50D | 200-Day MA | vs 200D | Trend |`);
ln(`|-------|--------|------:|----------:|-------:|-----------:|--------:|-------|`);
for (const q of equities) {
  const vs50  = parseFloat(maPct(q.price, q.priceAvg50).replace('%',''));
  const vs200 = parseFloat(maPct(q.price, q.priceAvg200).replace('%',''));
  const trend = vs50 > 0 && vs200 > 0 ? '**Bullish**' : vs50 < 0 && vs200 < 0 ? '**Bearish**' : 'Mixed';
  ln(`| **${q.label}** | ${q.region} | ${fmt(q.price)} | ${fmt(q.priceAvg50)} | ${vs50 >= 0 ? '+' : ''}${vs50.toFixed(2)}% | ${fmt(q.priceAvg200)} | ${vs200 >= 0 ? '+' : ''}${vs200.toFixed(2)}% | ${trend} |`);
}

// 6. Regional summary
h('6. Regional Summary', 2);
for (const region of ['US', 'Europe', 'Asia']) {
  const group = equities.filter(q => q.region === region);
  const avgChg = (group.reduce((s, q) => s + (q.changePercentage ?? 0), 0) / group.length);
  ln(`\n**${region}** — avg change: ${avgChg >= 0 ? '+' : ''}${avgChg.toFixed(2)}%`);
  ln(`| Index | Price | Chg % |`);
  ln(`|-------|------:|------:|`);
  for (const q of group) {
    ln(`| ${q.label} | ${fmt(q.price)} | ${pct(q.changePercentage)} |`);
  }
}

hr();
ln(`*Data source: Financial Modeling Prep /stable/quote. Timestamps reflect last exchange close for each region.*`);

// ── write file ────────────────────────────────────────────────────────────────

const output = lines.join('\n');
const filename = `indices-${dateSlug}.md`;
writeFileSync(filename, output, 'utf8');
process.stdout.write(output + '\n');
process.stderr.write(`\nSaved → ${filename}\n`);

// ── bar chart helper ──────────────────────────────────────────────────────────

function buildBar(price, low, high, width) {
  if (price == null || high <= low) return '?'.repeat(width);
  const pos = Math.round(((price - low) / (high - low)) * (width - 1));
  return '░'.repeat(pos) + '█' + '░'.repeat(width - 1 - pos);
}
