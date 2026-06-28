# Agent Response — sec-filings-analyst

**Query:** What are the most significant 8-K filings from the past 30 days and what do they signal for investors?

**Agent:** sec-filings-analyst
**Model:** nvidia/nemotron-3-ultra-550b-a55b:free
**Date:** 2026-06-28

---

I don't have real-time access to SEC EDGAR or live filing feeds. My knowledge is current only through my training cutoff, and I cannot pull the past 30 days of 8-K filings.

---

## What I Can Do Instead

### If You Provide Filing Data
Give me a list of recent 8-Ks (ticker, date, item numbers, headline) and I will:
- Triage by materiality using the framework in my instructions
- Flag Tier 1/2/3 signals
- Quantify impact where figures are disclosed
- Note cross-filing patterns or sector clusters
- Recommend next steps (deep-dive, monitor, trade catalyst, ignore)

### If You Need Real-Time Coverage, Use These Sources

| Tool | Strength | Cost |
|------|----------|------|
| **SEC EDGAR RSS/Atom feeds** | Free, official, complete | Free |
| **Benzinga Pro / TradeTheNews** | Real-time headlines, item tagging | $$ |
| **FactSet / CapIQ / Bloomberg** | Full-text search, exhibits, analytics | $$$$ |
| **EDGAR Dashboard (WhaleWisdom, EDGAR Online)** | Filtered alerts by item, sector, cap | $–$$ |
| **Twitter/X: @SEC_Enforcement, @EdgarAlert** | Fast headlines, not comprehensive | Free |
| **Quiver Quantitative / OpenInsider** | Unusual activity screens | Free / $ |

### Practical Workflow You Can Run Today
1. **Pull last 30 days** via EDGAR API or your terminal: `formType=8-K AND filingDate>=2025-07-16`
2. **Filter to high-signal items**: 1.01, 1.03, 2.01, 2.04, 2.06, 5.02, 5.07
3. **Exclude routine**: 7.01 (most presentations), 8.01 (read individually), 9.01 (exhibit list only)
4. **Screen for materiality**: deal size >5% market cap, CEO/CFO departure, impairment >10% equity, covenant breach
5. **Send me the shortlist** — I'll interpret the investment signal.

---

## Recent High-Profile Patterns (from training data, not live)

| Pattern | Typical 8-K Items | Investor Takeaway |
|---------|-------------------|-------------------|
| **M&A wave in semiconductors** | 1.01 → 2.01 chain | Acquirers often sell off on close; targets gap up then drift |
| **Biotech pipeline setbacks** | 2.06 (impairment) + 8.01 (trial failure) | Stock often overshoots; watch for 5.02 (CSO departure) confirming |
| **Regional bank stress** | 2.03 (FHLB draws), 2.04 (covenant), 5.02 (CRO exit) | Equity impaired; debt/restructuring equities become the trade |
| **Activist campaigns** | 5.08 (nominations), 5.07 (vote), 1.01 (cooperation agreement) | 60-day window from 13D to 5.07 is the alpha period |

---

**Bottom line**: Send me a curated list of 10–20 filings (ticker, date, items, one-line summary) and I'll give you the institutional-grade signal extraction you're looking for.
