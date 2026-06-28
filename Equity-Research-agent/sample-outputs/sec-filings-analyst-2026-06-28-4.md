# Agent Response — sec-filings-analyst

**Query:** I want to screen for activist investor activity. Walk me through how to use 13D filings to identify hedge funds accumulating stakes, and what to look for in the Schedule 13D to assess whether an activist campaign is likely.

**Agent:** sec-filings-analyst
**Model:** poolside/laguna-m.1:free
**Date:** 2026-06-28

---


# Activist Investor Screening Using 13D Filings

## Understanding Form 13D — The Activist's Calling Card

Form 13D is filed when any person or group acquires beneficial ownership of **5% or more** of any class of equity securities registered under Section 12 of the Exchange Act. It's your window into activist accumulation before they've even announced their intentions.

### Key Thresholds to Track
- **5% threshold**: Initial trigger for 13D filing requirement
- **10% threshold**: Regulatory spotlight intensifies; management attention guaranteed
- **20% threshold**: Control becomes possible; board seat negotiations often begin
- **30% threshold**: Proxy solicitation rights triggered under Rule 14a

## Screening Methodology

### 1. Data Sources & Access Points
```
Primary Sources:
• SEC EDGAR database (sec.gov/edgar)
• Form 13D RSS feeds (real-time)
• Commercial providers: Bloomberg, FactSet, Refinitiv, S&P Capital IQ
• Specialized tools: SharkRepellent, ProxyInsight, Sentieo

Screening Parameters:
• Filter for "Schedule 13D" (not 13G)
• Group formations (indicates coordinated activism)
• Filing dates within last 30-60 days
• Equity positions >$50M (institutional quality threshold)
```

### 2. High-Conviction Patterns to Flag

**Pattern #1 — Group Formation Immediately Below Thresholds**
- Multiple 13D filings (same target, different activists) within days of each other
- Often indicates pre-negotiated consortium approach
- Example: Starboard Value + Legion Partners both filing within 48 hours = coordinated campaign likely

**Pattern #2 — Rapid Accumulation Above 10%**
- Original filing at 5.x% followed by amendments within 2-3 weeks showing 12-15% positions
- Suggests significant open market buying or derivative structuring
- Monitor amendments closely—activists often under-disclose initially

**Pattern #3 — Track Record Clustering**
- Same activist (Pershing Square, Elliott, Starboard) filing on multiple targets in same quarter
- Sector concentration suggests opportunistic thematic play
- Review their past 12-month campaign success rate

## Deep Dive Analysis — What the Schedule 13D Reveals

### Critical Sections to Exhume:

**Item A — Identity of Person Filing**
- **Experience level**: Partnership formation date, assets under management
- **Activist history**: Previous 13D/13G filings, track record of board seats won
- **Group composition**: If filing as group, who else is involved and their respective stakes

**Item B — Aggregate Amount and Percentage Owned**
- **Position size evolution**: Look for amendment patterns showing continued accumulation
- **Derivative exposure**: Options, swaps, total return swaps that may not be fully reflected in cash ownership
- **Economic interest vs voting power**: Critical for understanding true influence

**Item C — Purpose of Transaction**
- **This is the money shot**: Read every word carefully
  
  **Red flags for short-term activism:**
  - "We intend to engage with management regarding strategic alternatives"
  - "We believe the company's valuation does not reflect intrinsic value"
  - "Potential corporate restructuring, sale, or merger"

  **Traditional activist language:**
  - "Seek board representation or influence"
  - "Evaluate all options including operational changes"
  - "Believe current strategy undervalues shareholders"

**Item D — Equity Securities of Issuer (Trading Details)**
- **Average cost basis**: Compare to current trading—indicates paper profit/loss
- **Purchase timing**: Concentrated purchases suggest conviction play
- **Derivative instruments**: Often the real story—total return swaps can amplify exposure without showing full cash cost

### 3. Quantitative Signals to Calculate

**Influence Ratio**: `(Economic Interest + Derivatives) / Market Capitalization`
- >2% = Significant activist exposure warranting attention
- >5% = Campaign almost certainly underway

**Accumulation Velocity**: Days between original filing and peak disclosed ownership
- <30 days = Aggressive accumulation, high conviction
- >90 days = Likely stealth building or passive approach

**Burn Rate Assessment**: Implied funding needs if activist pursues proxy contest
- Typical proxy fight costs: $3-10M depending on market cap
- Cash on hand + position unrealized gains = runway for activism

## Cross-Reference Intelligence

### Link 13D to 8-K Filings:
- **Item 5.08** (shareholder director nominations) often follows 13D by 30-60 days
- **Item 7.01** (Reg FD disclosure) may contain investor presentation materials
- Watch for sudden **Item 5.02** (officer departures) after 13D filing

### Peer Group Context:
If Company A attracts activism, check:
- Peers trading at similar valuation discounts
- Companies with same operational weaknesses (ROIC, margins, capital allocation)
- Previous activist targets that successfully re-rated

## Alert Framework for Investment Teams

### Tier 1 Alerts (Immediate Action):
- Group formation filing with >10% combined stake
- Activist with >$2B AUM filing on sub-$1B market cap company
- Multiple activists filing on same target within 30 days

### Tier 2 Alerts (Monitor Closely):
- First-time activist filing on any target
- Position increasing 50%+ via amendment within 60 days
- Activist with 75%+ success rate in past campaigns shows interest

### Tier 3 Monitoring:
- Individual filing below 5% (may indicate larger positioning through derivatives)
- Filing by traditionally passive investors (could signal value realization)

## Common Pitfalls & Blind Spots

**Under-disclosure risk**: Activists sometimes file 13G instead of 13D to avoid scrutiny, particularly when using derivatives.

**Timing traps**: 13D amendments can lag actual accumulation by weeks—markets may already be pricing in higher positions.

**Passive positioning**: Some "activist" funds file 13D after taking significant positions but state passive intent—this often evolves into activism.

Would you like me to walk through a specific 13D filing you've identified, or discuss how to integrate this screening with other SEC forms for comprehensive activist detection?

