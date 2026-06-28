# SEC Filings — Security Types Reference

When pulling SEC filings via FMP, a single company (identified by its **CIK**) may have multiple listed securities — each with its own ticker. FMP returns one row per listed ticker, so the same 8-K will appear multiple times if the filer has both common stock and warrants on the same exchange.

---

## Real example — 2026-06-28 8-K batch

From `sample-outputs/sec-filings-8k-2026-06-28.json`:

| Ticker | CIK | Security type | Company |
|--------|-----|---------------|---------|
| `OTLK` | 0001649989 | Common stock | Outlook Therapeutics, Inc. |
| `OTLKW` | 0001649989 | Warrant | Outlook Therapeutics, Inc. |
| `LGL` | 0000061004 | Common stock | LGL Group, Inc. |
| `LGL-WT` | 0000061004 | Warrant | LGL Group, Inc. |
| `GRAF-UN` | 0001897463 | Unit (SPAC) | Graf Acquisition Corp. |

Both pairs (OTLK/OTLKW and LGL/LGL-WT) share a CIK because they are the same legal entity — one company, two classes of security. The SEC assigns one CIK per registrant regardless of how many securities it has listed.

---

## Security type definitions

### Common stock

Standard equity ownership in the company. Shareholders have voting rights, residual claim on assets, and participate in dividends and buybacks. The base security — every publicly traded company has it.

- **Ticker convention:** plain ticker, no suffix (`OTLK`, `LGL`)

---

### Warrant

A derivative security issued **directly by the company** (unlike exchange-traded options, which are created by the exchange). Gives the holder the right — but not the obligation — to buy common shares at a fixed price before an expiry date.

| Feature | Detail |
|---------|--------|
| **Exercise price** | Fixed strike at which the holder can buy common shares. Warrant has intrinsic value only when the stock trades *above* this level. |
| **Expiry** | Finite life — expires worthless if the stock never exceeds the exercise price. |
| **Dilution** | When exercised, the company issues *new* shares. This dilutes existing shareholders, which is why warrants are disclosed and tracked separately from common stock. |
| **Leverage** | Warrants trade at a fraction of the common share price but move directionally with it — amplified upside, total loss if stock stays below exercise price. |

- **Ticker convention:** `-W` or `-WT` suffix (`OTLKW`, `LGL-WT`)

---

### Unit (SPAC)

Common in **Special Purpose Acquisition Companies (SPACs)**. A unit bundles a common share and one or more warrants into a single tradeable security at IPO. Units typically separate (split) into their component parts 52 days after the SPAC IPO, after which common shares and warrants trade independently.

- **Ticker convention:** `-UN` or `-U` suffix (`GRAF-UN`)
- **Implication for filings:** A SPAC filing an 8-K before its unit split will show the `-UN` ticker. Post-split, the same entity files under the common stock and `-W` warrant tickers separately.

---

## Ticker suffix reference

| Suffix | Security type | Typical context |
|--------|---------------|-----------------|
| *(none)* | Common stock | All listed companies |
| `-W` / `-WT` | Warrant | Post-SPAC, biotech raises, structured deals |
| `-UN` / `-U` | Unit (share + warrant bundle) | SPAC pre-split |
| `-R` | Right | Short-dated right to subscribe to new shares |
| `-P` / `PRA` / `PRB` | Preferred share (series A, B…) | Banks, utilities, hybrid capital |

---

## Why the same 8-K appears multiple times

The SEC's EDGAR system registers filings against a **CIK** (Central Index Key), not a ticker. One 8-K filing covers the entire legal entity. FMP maps each filing to every ticker associated with that CIK — so if a company has common stock, warrants, and a preferred series all listed, the same 8-K document will appear as three rows in the API response.

**When deduplicating:** group by `cik` + `acceptedDate` + `link`. If all three match, the rows describe the same filing document across different security classes.

---

## Related files

- `sample-outputs/sec-filings-8k-2026-06-28.json` — live 8-K response example (5 filings, 2026-06-28)
- `agents/sec-filings-analyst.md` — SEC filings analyst agent definition
- `api.py` — `/sec-filings`, `/sec-filings/form-type`, `/sec-filings/search` endpoints
