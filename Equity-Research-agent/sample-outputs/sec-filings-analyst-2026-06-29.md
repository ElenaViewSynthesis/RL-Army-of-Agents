# Agent Response — sec-filings-analyst

**Query:** I want to screen for activist investor activity. Walk me through how to use 13D filings to identify hedge funds accumulating stakes, and what to look for in the Schedule 13D to assess whether an activist campaign is likely.

**Agent:** sec-filings-analyst
**Model:** poolside/laguna-m.1:free
**Date:** 2026-06-29

---


# Activist Investor Screening via Schedule 13D

The 13D filing is your primary early detection system for activist campaigns. Unlike the 8-K (which reports events after they happen), 13D gives you advance notice of accumulation and intent. Here's how to weaponize it:

## Accessing and Screening 13D Filings

### Primary Sources:
- **SEC EDGAR** → Search "Form 13D" or use RSS feeds for real-time alerts
- **Bloomberg Terminal** → SECFL <GO> for 13D monitoring
- **S&P Capital IQ**, **FactSet** → Built-in activist screens
- **WhaleWisdom**, **Fintel.io** → Free/paid services aggregating 13D/D filings

### Screening Filters:
- **New 13D filings within last 30 days** (fresh accumulation)
- **Ownership >5%** threshold crossed (material enough to require filing)
- **Hedge fund identifiable holders** (look up investor history)
- **Purpose section contains activist language** (screen for keywords below)
- **Concurrent 13F activity** (confirms institutional accumulation pattern)

## Anatomy of Schedule 13D — Activist Signals

### Section 2: "Purpose of Transaction" (CRITICAL)
This is where activists telegraph their playbook:

**Explicit Activist Language (Campaign Likely):**
- "We intend to seek board representation" 
- "Propose changes to management/strategy"
- "Explore merger or sale of the company"
- "Seek to acquire additional shares"
- "Engage with management regarding operations"

**Passive Language (Probably Not an Activist):**
- "Long-term investment in ordinary course"
- "Trading for investment purposes"
- "May acquire additional shares opportunistically"

**Red Flag Phrasing:**
- "Acquired shares for liquidity event" (often means they're positioning for a sale)
- "Intend to communicate with other shareholders" (coordination signal)

### Section 3: "Source and Amount of Funds"
Activists often reveal their war chest here:
- Large committed capital suggests sustained campaign
- "Committed financing" indicates ability to maintain pressure
- Multiple funding sources suggest sophisticated campaign backing

### Section 4: "Relationships to Issuer"
Cross-reference with company filings:
- **No prior relationship** = typical activist entry
- **Former employee/affiliate** = reconquest attempt
- **Business partner** = strategic angle, not pure activism

### Section 5: "Agreements, Understandings, or Plans"
This is gold for predicting next steps:
- Detailed plans with timelines = organized campaign
- References to "other shareholders" = coalition building
- Specific strategic proposals = higher probability of forcing change

## Activist Campaign Probability Matrix

### High Probability Activist (Tradeable Signal):
| Factor | Signal |
|--------|--------|
| Ownership | 5-15% range crossing |
| Purpose section | Contains board strategy language |
| History | Holder has activist track record |
| Group status | Indicates coordination with others |
| Timeline | Multiple purchases in 30-day window |
| Source of funds | Committed capital disclosed |

### Monitor/Tail Situation:
| Factor | Signal |
|--------|--------|
| Ownership | >15% (may seek control) |
| Purpose section | Vague or "investment only" |
| Holdings pattern | Concentrated position |
| Previous exits | History of quick exits after small gains |

## Cross-Filing Pattern Analysis

### 13D + 8-K Correlation:
Activist campaigns often generate multiple filing types:
- **13D filing** → Initial position disclosure
- **8-K Item 5.02** → Executive departures (management shakeup)
- **8-K Item 5.07** → Shareholder meeting votes (momentum building)
- **13D amendments** → Increasing positions (doubling down)

### Red Flags in Subsequent Filings:
- **8-K Item 1.01** with activist mentioned as counterparty (M&A transaction)
- **8-K Item 3.03** (rights modifications) after activist accumulation
- **Multiple CFO exits** within 6 months (strategy shift indication)

## Notable Activist Hedge Funds to Track

Based on filing patterns and historical outcomes:
- **Starboard Value** (typically declares specific strategic plans)
- **Pershing Square** (large positions, detailed presentations)
- **Elliott Management** (litigation-oriented, explicit change demands)
- **Third Point** (operational expertise claims in filings)
- **Carl Icahn** (historically aggressive language in Purpose section)

## Timing Signals

### Entry Patterns:
- **Stealth accumulation** (below 5% threshold) followed by 13D filing
- **Quarter-end clustering** (institutional window dressing)
- **Post-earnings weakness** (buying on bad news)

### Escalation Indicators:
- **Amendment #2 or #3** to original 13D (position increase)
- **Concurrent 13F increase** (public confirmation)
- **Media coverage spike** matching filing date (intentional signaling)

## What to Do With This Information

### Immediate Actions:
1. **Cross-check the investor's track record** on prior 13Ds
2. **Parse the Exhibit 99** (investor presentation) for specific proposals
3. **Monitor the next earnings call** for management acknowledgment/response
4. **Watch for related 8-K filings** within 30-60 days

### Trade Considerations:
- **Long side**: Companies where activist identifies undervaluation AND has operational plan
- **Short side**: Companies facing activists without clear strategic alternative
- **Volatility trades**: Pre-vote situations (Item 5.07 outcomes) often create binary moves

### Risk Assessment:
- Has the market already priced in the activist's goals?
- Does management have a credible counter-strategy?
- Any regulatory/governance barriers to proposed changes?

The key insight: 13D filings reveal both conviction AND capability. An activist with detailed plans, committed capital, and a history of forcing change represents a genuine catalyst. One with vague language and no track record may just be a large holder waiting for someone else to do the work.

