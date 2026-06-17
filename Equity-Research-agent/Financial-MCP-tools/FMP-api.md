# FMP MCP Tools Reference

Complete reference for the 27 Financial Modeling Prep (FMP) MCP tools available in Claude Code. All tools share the same call pattern: one required `endpoint` parameter plus optional parameters that vary per endpoint.

**Plan requirements** are noted where they apply. Free tier allows 250 requests/day.

---

## Table of Contents

| # | Tool | Category |
|---|------|----------|
| 1 | [company](#1-company) | Company fundamentals & corporate data |
| 2 | [statements](#2-statements) | Financial statements & ratios |
| 3 | [quote](#3-quote) | Real-time price quotes |
| 4 | [analyst](#4-analyst) | Analyst ratings & price targets |
| 5 | [discountedCashFlow](#5-discountedcashflow) | DCF valuation models |
| 6 | [insiderTrades](#6-insidertrades) | Insider trading disclosures |
| 7 | [news](#7-news) | Financial news & press releases |
| 8 | [technicalIndicators](#8-technicalindicators) | Technical analysis indicators *(Starter+)* |
| 9 | [earningsTranscript](#9-earningstranscript) | Earnings call transcripts *(Ultimate+)* |
| 10 | [secFilings](#10-secfilings) | SEC filings |
| 11 | [search](#11-search) | Symbol search & stock screener |
| 12 | [calendar](#12-calendar) | Earnings, dividends, IPO & split calendars |
| 13 | [ESG](#13-esg) | ESG ratings *(Ultimate+)* |
| 14 | [form13F](#14-form13f) | Institutional ownership (13F) *(Ultimate+)* |
| 15 | [senate](#15-senate) | Congressional trading disclosures |
| 16 | [chart](#16-chart) | Historical & intraday price charts |
| 17 | [indexes](#17-indexes) | Market indexes (S&P 500, Nasdaq, DJIA) |
| 18 | [marketPerformance](#18-marketperformance) | Sector/industry performance snapshots |
| 19 | [commodity](#19-commodity) | Commodity market data |
| 20 | [crypto](#20-crypto) | Cryptocurrency market data |
| 21 | [forex](#21-forex) | Forex currency pair data |
| 22 | [etfAndMutualFunds](#22-etfandmutualfunds) | ETF & mutual fund analysis |
| 23 | [economics](#23-economics) | Macroeconomic indicators |
| 24 | [marketHours](#24-markethours) | Exchange hours & holiday schedules |
| 25 | [Fundraisers](#25-fundraisers) | Crowdfunding & equity offerings |
| 26 | [commitmentOfTraders](#26-commitmentoftraders) | CFTC COT reports *(Premium+)* |
| 27 | [directory](#27-directory) | Reference lists & symbol directories |

---

## 1. `company`

Company fundamentals and corporate data: profiles, executives, M&A activity, employee counts, market cap, shares float, and peer comparisons.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `profile-symbol` | `symbol` | Full company profile (sector, industry, CEO, description, website) |
| `profile-cik` | `cik` | Company profile lookup by SEC CIK number |
| `company-executives` | `symbol` | List of executives with titles and compensation |
| `executive-compensation` | `symbol` | Detailed exec compensation breakdown |
| `executive-compensation-benchmark` | — | Benchmark exec compensation across the market |
| `employee-count` | `symbol` | Current employee headcount |
| `historical-employee-count` | `symbol` | Employee count over time |
| `market-cap` | `symbol` | Current market capitalization |
| `historical-market-cap` | `symbol` | Historical market cap time series |
| `batch-market-cap` | `symbols` | Market cap for multiple symbols at once |
| `shares-float` | `symbol` | Float shares and liquidity metrics |
| `all-shares-float` | — | Float data for all listed companies |
| `peers` | `symbol` | Peer/competitor companies in the same sector |
| `company-notes` | `symbol` | Company disclosures and footnotes |
| `latest-mergers-acquisitions` | — | Recent M&A activity across the market |
| `search-mergers-acquisitions` | `name` | Search M&A deals by company name |
| `delisted-companies` | — | List of delisted companies |

**Parameters:** `symbol`, `symbols` (array), `cik`, `name`, `limit`, `page`, `from_date`, `to_date`, `year`

**Example:**
```
tool: company
endpoint: profile-symbol
symbol: AAPL
```

---

## 2. `statements`

Financial statements and ratios: income statements, balance sheets, cash flow statements — standard, as-reported, TTM, and growth rates. Also key metrics, financial ratios, enterprise values, owner earnings, and revenue segmentation.

Use `period: "annual"` or `"quarter"`. Some endpoints also accept `"FY"`, `"Q1"`–`"Q4"`.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `income-statement` | `symbol` | Annual/quarterly income statement |
| `income-statements-ttm` | `symbol` | Trailing twelve months income statement |
| `income-statement-growth` | `symbol` | YoY income statement growth rates |
| `balance-sheet-statement` | `symbol` | Annual/quarterly balance sheet |
| `balance-sheet-statements-ttm` | `symbol` | TTM balance sheet |
| `balance-sheet-statement-growth` | `symbol` | YoY balance sheet growth rates |
| `cashflow-statement` | `symbol` | Annual/quarterly cash flow statement |
| `cashflow-statements-ttm` | `symbol` | TTM cash flow statement |
| `cashflow-statement-growth` | `symbol` | YoY cash flow growth rates |
| `key-metrics` | `symbol` | Key metrics per period (P/E, EV/EBITDA, ROE, FCF yield…) |
| `key-metrics-ttm` | `symbol` | TTM key metrics |
| `metrics-ratios` | `symbol` | Financial ratios per period (margins, liquidity, solvency) |
| `metrics-ratios-ttm` | `symbol` | TTM financial ratios |
| `financial-statement-growth` | `symbol` | Combined financial statement growth rates |
| `financial-scores` | `symbol` | Altman Z-Score, Piotroski F-Score, ESG and other scores |
| `enterprise-values` | `symbol` | Enterprise value components over time |
| `owner-earnings` | `symbol` | Buffett-style owner earnings calculation |
| `revenue-geographic-segments` | `symbol` | Revenue breakdown by geography |
| `revenue-product-segmentation` | `symbol` | Revenue breakdown by product/segment |
| `as-reported-income-statements` | `symbol` | Income statement as filed with SEC |
| `as-reported-balance-statements` | `symbol` | Balance sheet as filed with SEC |
| `as-reported-cashflow-statements` | `symbol` | Cash flow as filed with SEC |
| `as-reported-financial-statements` | `symbol` | All financials as filed with SEC |
| `financial-reports-dates` | `symbol` | Available 10-K filing dates |
| `financial-reports-form-10-k-json` | `symbol`, `year`, `period` | Full 10-K report in JSON format |
| `financial-reports-form-10-k-xlsx` | `symbol`, `year`, `period` | Full 10-K report as XLSX |
| `latest-financial-statements` | — | Most recent financial filings across all companies |

**Parameters:** `symbol`, `period`, `limit`, `page`, `year`, `structure`

**Example:**
```
tool: statements
endpoint: income-statement
symbol: MSFT
period: annual
limit: 4
```

---

## 3. `quote`

Real-time and aftermarket stock quotes — single, batch, and full-exchange sweeps. Also covers ETF, mutual fund, crypto, commodity, forex, and index quotes.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `quote` | `symbol` | Full real-time quote (price, volume, market cap, PE, 52W range) |
| `quote-short` | `symbol` | Lightweight quote (price, volume, change) |
| `quote-change` | `symbol` | Price change over 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, 10Y |
| `batch-quote` | `symbols` | Full quotes for multiple symbols |
| `batch-quote-short` | `symbols` | Lightweight quotes for multiple symbols |
| `aftermarket-quote` | `symbol` | After-hours / pre-market quote |
| `aftermarket-trade` | `symbol` | Most recent aftermarket trade |
| `batch-aftermarket-quote` | `symbols` | Aftermarket quotes for multiple symbols |
| `batch-aftermarket-trade` | `symbols` | Aftermarket trades for multiple symbols |
| `full-exchange-quotes` | `exchange` | All stock quotes for an entire exchange |
| `full-etf-quotes` | — | All ETF quotes |
| `full-mutualfund-quotes` | — | All mutual fund quotes |
| `full-cryptocurrency-quotes` | — | All crypto quotes |
| `full-forex-quotes` | — | All forex pair quotes |
| `full-commodities-quotes` | — | All commodity quotes |
| `full-index-quotes` | — | All index quotes |

**Parameters:** `symbol`, `symbols` (array), `exchange`, `short`

**Example:**
```
tool: quote
endpoint: batch-quote
symbols: ["AAPL", "MSFT", "GOOGL"]
```

---

## 4. `analyst`

Analyst ratings, price targets, and financial estimates. Covers current and historical grades, consensus price targets, and EPS/revenue estimates.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `grades` | `symbol` | Recent analyst grades from investment banks |
| `grades-summary` | `symbol` | Summary count of Buy/Hold/Sell grades |
| `historical-grades` | `symbol` | Historical analyst grade history |
| `ratings-snapshot` | `symbol` | Current ratings snapshot (DCF, ROE, PB, PS, PEG based) |
| `historical-ratings` | `symbol` | FMP proprietary historical rating scores |
| `price-target-consensus` | `symbol` | Consensus price target (high, low, median, average) |
| `price-target-summary` | `symbol` | Price target summary with analyst count |
| `financial-estimates` | `symbol`, `period` | Consensus EPS and revenue estimates |

**Parameters:** `symbol`, `period` (`"annual"` or `"quarter"`), `limit`, `page`

**Example:**
```
tool: analyst
endpoint: price-target-consensus
symbol: NVDA
```

---

## 5. `discountedCashFlow`

DCF valuation models: standard and levered, with optional custom assumption overrides (growth rate, WACC components, margins, etc.).

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `dcf-advanced` | `symbol` | Standard DCF intrinsic value vs. current price |
| `dcf-levered` | `symbol` | Levered DCF (accounts for debt in valuation) |
| `custom-dcf-advanced` | `symbol` | DCF with custom assumption overrides |
| `custom-dcf-levered` | `symbol` | Levered DCF with custom assumption overrides |

**Custom assumption parameters** (all optional strings, expressed as decimals e.g. `"0.05"` for 5%):

| Parameter | Description |
|-----------|-------------|
| `revenueGrowthPct` | Projected revenue growth rate |
| `ebitdaPct` | EBITDA as % of revenue |
| `ebitPct` | EBIT as % of revenue |
| `taxRate` | Effective tax rate |
| `longTermGrowthRate` | Terminal growth rate |
| `riskFreeRate` | Risk-free rate (e.g. 10Y Treasury yield) |
| `marketRiskPremium` | Equity risk premium |
| `beta` | Company beta |
| `costOfDebt` | Pre-tax cost of debt |
| `costOfEquity` | Cost of equity (overrides CAPM calc) |
| `operatingCashFlowPct` | OCF as % of revenue |
| `capitalExpenditurePct` | Capex as % of revenue |
| `depreciationAndAmortizationPct` | D&A as % of revenue |
| `sellingGeneralAndAdministrativeExpensesPct` | SG&A as % of revenue |
| `cashAndShortTermInvestmentsPct` | Cash as % of revenue |
| `receivablesPct` | Receivables as % of revenue |
| `inventoriesPct` | Inventories as % of revenue |
| `payablePct` | Payables as % of revenue |

**Example:**
```
tool: discountedCashFlow
endpoint: custom-dcf-advanced
symbol: TSLA
revenueGrowthPct: "0.20"
longTermGrowthRate: "0.03"
taxRate: "0.21"
```

---

## 6. `insiderTrades`

Insider trading data from SEC Form 4 filings: executive purchases and sales, acquisition ownership changes, trade statistics, and searchable by name or CIK.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `latest-insider-trade` | — | Most recent insider transactions across all companies |
| `search-insider-trades` | — | Search trades by symbol, CIK, or transaction type |
| `insider-trade-statistics` | `symbol` | Insider buy/sell statistics (net activity, value) |
| `acquisition-ownership` | `symbol` | 5%+ ownership acquisition filings |
| `search-reporting-name` | `name` | Find all trades by a specific insider's name |
| `all-transaction-types` | — | Reference list of all valid transaction type codes |

**Parameters:** `symbol`, `companyCik`, `reportingCik`, `transactionType`, `date`, `name`, `limit`, `page`

**Example:**
```
tool: insiderTrades
endpoint: search-insider-trades
symbol: META
limit: 20
```

---

## 7. `news`

Financial news and press releases: stock-specific, market-wide, crypto, forex, and FMP editorial articles. Filterable by date range and symbol list.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `search-stock-news` | `symbols` | News articles for specific stock tickers |
| `search-press-releases` | `symbols` | Press releases for specific tickers |
| `stock-news` | — | General stock market news feed |
| `general-news` | — | Broad market and financial news |
| `press-releases` | — | All recent press releases |
| `fmp-articles` | — | FMP editorial and analysis articles |
| `search-crypto-news` | `symbols` | News for specific crypto assets |
| `crypto-news` | — | General crypto news feed |
| `search-forex-news` | `symbols` | News for specific forex pairs |
| `forex-news` | — | General forex news feed |

**Parameters:** `symbols` (array), `from_date`, `to_date`, `limit`, `page`

**Example:**
```
tool: news
endpoint: search-stock-news
symbols: ["AAPL", "MSFT"]
limit: 10
from_date: "2026-06-01"
```

---

## 8. `technicalIndicators`

*(Requires Starter, Premium, Ultimate, or Enterprise plan)*

Technical analysis indicators across multiple timeframes. All endpoints require `symbol`, `periodLength`, and `timeframe`.

**Timeframes:** `1min`, `5min`, `15min`, `30min`, `1hour`, `4hour`, `1day`

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `simple-moving-average` | SMA — average price over N periods |
| `exponential-moving-average` | EMA — weighted average, recent prices weighted more |
| `double-exponential-moving-average` | DEMA — reduces lag vs. EMA |
| `triple-exponential-moving-average` | TEMA — minimal lag moving average |
| `weighted-moving-average` | WMA — linearly weighted moving average |
| `relative-strength-index` | RSI — momentum oscillator (0–100 scale) |
| `average-directional-index` | ADX — trend strength indicator |
| `williams` | Williams %R — momentum oscillator |
| `standard-deviation` | Statistical volatility measure |

**Parameters:** `symbol`, `periodLength` (integer), `timeframe`, `from_date`, `to_date`

**Example:**
```
tool: technicalIndicators
endpoint: relative-strength-index
symbol: SPY
periodLength: 14
timeframe: 1day
```

---

## 9. `earningsTranscript`

*(Requires Ultimate or Enterprise plan)*

Full text of earnings call transcripts, searchable by symbol, year, and quarter.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `search-transcripts` | `symbol`, `year`, `quarter` | Full transcript text for a specific call |
| `transcripts-dates-by-symbol` | `symbol` | All available transcript dates for a company |
| `latest-transcripts` | — | Most recent transcripts across all companies |
| `available-transcript-symbols` | — | All symbols that have transcripts available |

**Parameters:** `symbol`, `year`, `quarter` (1–4), `limit`, `page`

**Example:**
```
tool: earningsTranscript
endpoint: search-transcripts
symbol: AAPL
year: 2025
quarter: 4
```

---

## 10. `secFilings`

SEC filings database: search by symbol, CIK, form type, or company name. Covers all form types (10-K, 10-Q, 8-K, etc.) plus SIC industry classification data.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `search-by-symbol` | `symbol` | All SEC filings for a ticker |
| `search-by-cik` | `cik` | All SEC filings by CIK number |
| `search-by-form-type` | `formType` | All filings of a specific form type (e.g. `"10-K"`) |
| `search-by-name` | `company` | Search filings by company name |
| `8k-latest` | — | Most recent 8-K filings (material events) |
| `financials-latest` | — | Most recent financial filings (10-K, 10-Q) |
| `company-search-by-symbol` | `symbol` | SEC company record by ticker |
| `company-search-by-cik` | `cik` | SEC company record by CIK |
| `sec-company-full-profile` | `symbol` | Full SEC company profile with filing history |
| `industry-classification-list` | — | All SIC industry classifications |
| `industry-classification-search` | — | Search SIC codes by `sicCode` or `industryTitle` |
| `all-industry-classification` | — | Complete SIC industry classification dataset |

**Parameters:** `symbol`, `cik`, `formType`, `company`, `sicCode`, `industryTitle`, `from_date`, `to_date`, `limit`, `page`

**Example:**
```
tool: secFilings
endpoint: search-by-form-type
formType: 10-K
from_date: "2025-01-01"
limit: 10
```

---

## 11. `search`

Symbol search and stock screener. Find any security by symbol, name, CIK, ISIN, or CUSIP. Screen stocks using financial and fundamental filters.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `search-symbol` | `query` | Search by ticker symbol |
| `search-name` | `query` | Search by company name |
| `search-CIK` | `cik` | Look up company by CIK |
| `search-ISIN` | `isin` | Look up security by ISIN |
| `search-cusip` | `cusip` | Look up security by CUSIP |
| `search-exchange-variants` | `symbol` | All exchange listings for a symbol |
| `search-company-screener` | — | Filter stocks by multiple criteria |

**Screener filter parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sector` | string | e.g. `"Technology"`, `"Healthcare"` |
| `industry` | string | e.g. `"Software"`, `"Semiconductors"` |
| `country` | string | e.g. `"US"`, `"CN"` |
| `exchange` | string | e.g. `"NYSE"`, `"NASDAQ"` |
| `marketCapMoreThan` | number | Min market cap (USD) |
| `marketCapLowerThan` | number | Max market cap (USD) |
| `priceMoreThan` | number | Min stock price |
| `priceLowerThan` | number | Max stock price |
| `betaMoreThan` | number | Min beta |
| `betaLowerThan` | number | Max beta |
| `volumeMoreThan` | number | Min average volume |
| `volumeLowerThan` | number | Max average volume |
| `dividendMoreThan` | number | Min dividend yield |
| `dividendLowerThan` | number | Max dividend yield |
| `isEtf` | boolean | Filter for ETFs only |
| `isFund` | boolean | Filter for funds only |
| `isActivelyTrading` | boolean | Filter for actively traded securities |
| `includeAllShareClasses` | boolean | Include all share classes |

**Example:**
```
tool: search
endpoint: search-company-screener
sector: Technology
marketCapMoreThan: 10000000000
country: US
isActivelyTrading: true
limit: 50
```

---

## 12. `calendar`

Market event calendars: upcoming and historical earnings reports, dividend payments, IPOs, and stock splits.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `earnings-calendar` | — | Upcoming earnings reports (market-wide) |
| `earnings-company` | `symbol` | Earnings history and upcoming date for a stock |
| `dividends-calendar` | — | Upcoming dividend ex-dates and payment dates |
| `dividends-company` | `symbol` | Dividend history for a specific stock |
| `ipos-calendar` | — | Upcoming IPO schedule |
| `ipos-prospectus` | — | IPO prospectus filings |
| `ipos-disclosure` | — | IPO disclosure documents |
| `splits-calendar` | — | Upcoming stock splits |
| `splits-company` | `symbol` | Historical stock splits for a company |

**Parameters:** `symbol`, `from_date`, `to_date`, `limit`, `page`, `includeReportTimes`

**Example:**
```
tool: calendar
endpoint: earnings-calendar
from_date: "2026-06-17"
to_date: "2026-06-30"
```

---

## 13. `ESG`

*(Requires Ultimate or Enterprise plan)*

Environmental, Social, and Governance (ESG) ratings and benchmarks.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `esg-ratings` | `symbol` | ESG scores for a specific company |
| `esg-search` | `symbol` | ESG investment data and search |
| `esg-benchmark` | `year` | Market-wide ESG benchmark comparison |

**Parameters:** `symbol`, `year`

**Example:**
```
tool: ESG
endpoint: esg-ratings
symbol: MSFT
```

---

## 14. `form13F`

*(Requires Ultimate or Enterprise plan)*

SEC Form 13F institutional ownership filings: extract holdings by fund CIK, view quarterly positions, analyze industry concentration, and track holder performance.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `latest-filings` | — | Most recent 13F filings across all institutions |
| `form-13f-filings-dates` | `cik` | All filing dates for a specific fund |
| `filings-extract` | `cik`, `year`, `quarter` | Full holdings list for a fund in a quarter |
| `filings-extract-with-analytics-by-holder` | `symbol`, `year`, `quarter` | All holders of a stock with analytics |
| `positions-summary` | `symbol`, `year`, `quarter` | Summary of institutional positions in a stock |
| `holder-performance-summary` | `cik` | Performance attribution for a fund's holdings |
| `holders-industry-breakdown` | `cik`, `year`, `quarter` | Industry concentration of a fund's holdings |
| `industry-summary` | `year`, `quarter` | Aggregate institutional ownership by industry |

**Parameters:** `symbol`, `cik`, `year`, `quarter`, `limit`, `page`

**Example:**
```
tool: form13F
endpoint: positions-summary
symbol: AAPL
year: 2025
quarter: 4
```

---

## 15. `senate`

U.S. congressional trading disclosures under the STOCK Act — Senate and House financial disclosure reports, searchable by stock or politician name.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `senate-latest` | — | Most recent Senate disclosure filings |
| `senate-trading` | `symbol` | Senate trades in a specific stock |
| `senate-trading-by-name` | `name` | All trades by a specific senator |
| `house-latest` | — | Most recent House disclosure filings |
| `house-trading` | `symbol` | House trades in a specific stock |
| `house-trading-by-name` | `name` | All trades by a specific representative |

**Parameters:** `symbol`, `name`, `limit`, `page`

**Example:**
```
tool: senate
endpoint: senate-trading
symbol: NVDA
```

---

## 16. `chart`

Historical end-of-day (EOD) and intraday price chart data. Supports dividend-adjusted, split-adjusted, and unadjusted series.

**EOD Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `historical-price-eod-full` | OHLCV + adjusted close (full dataset) |
| `historical-price-eod-light` | Date + close price only (compact) |
| `historical-price-eod-dividend-adjusted` | Dividend-adjusted prices |
| `historical-price-eod-non-split-adjusted` | Raw unadjusted prices |

**Intraday Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `intraday-1-min` | 1-minute OHLCV bars |
| `intraday-5-min` | 5-minute OHLCV bars |
| `intraday-15-min` | 15-minute OHLCV bars |
| `intraday-30-min` | 30-minute OHLCV bars |
| `intraday-1-hour` | 1-hour OHLCV bars |
| `intraday-4-hour` | 4-hour OHLCV bars |

**Parameters:** `symbol`, `from_date`, `to_date`, `nonadjusted` (boolean)

**Example:**
```
tool: chart
endpoint: historical-price-eod-full
symbol: SPY
from_date: "2024-01-01"
to_date: "2026-06-17"
```

---

## 17. `indexes`

Stock market index data: S&P 500, Nasdaq, and Dow Jones constituents, historical performance, and real-time quotes for any index symbol.

**Index constituent endpoints:**

| Endpoint | Description |
|----------|-------------|
| `sp-500` | Current S&P 500 constituent list |
| `historical-sp-500` | Historical S&P 500 membership changes |
| `nasdaq` | Current Nasdaq constituent list |
| `historical-nasdaq` | Historical Nasdaq changes |
| `dow-jones` | Current Dow Jones constituent list |
| `historical-dow-jones` | Historical Dow Jones changes |
| `indexes-list` | All available index symbols |

**Quote and chart endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `index-quote` | `symbol` | Real-time index quote |
| `index-quote-short` | `symbol` | Lightweight index quote |
| `all-index-quotes` | — | All index quotes at once |
| `index-historical-price-eod-full` | `symbol` | Full EOD history for an index |
| `index-historical-price-eod-light` | `symbol` | Light EOD history for an index |
| `index-intraday-1-min` | `symbol` | 1-minute intraday index bars |
| `index-intraday-5-min` | `symbol` | 5-minute intraday index bars |
| `index-intraday-1-hour` | `symbol` | 1-hour intraday index bars |

**Parameters:** `symbol`, `from_date`, `to_date`, `short`

**Example:**
```
tool: indexes
endpoint: sp-500
```

---

## 18. `marketPerformance`

Market-wide and sector/industry performance snapshots: movers, valuation multiples, and historical performance trends.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `biggest-gainers` | — | Top gaining stocks today |
| `biggest-losers` | — | Top losing stocks today |
| `most-active` | — | Most actively traded stocks by volume |
| `sector-performance-snapshot` | `date` | Sector performance on a specific date |
| `industry-performance-snapshot` | `date` | Industry performance on a specific date |
| `sector-PE-snapshot` | `date` | Sector P/E multiples on a specific date |
| `industry-PE-snapshot` | `date` | Industry P/E multiples on a specific date |
| `historical-sector-performance` | `sector` | Historical performance for a sector |
| `historical-industry-performance` | `industry` | Historical performance for an industry |
| `historical-sector-pe` | `sector` | Historical P/E ratio for a sector |
| `historical-industry-pe` | `industry` | Historical P/E ratio for an industry |

**Parameters:** `sector`, `industry`, `date`, `from_date`, `to_date`, `exchange`

**Example:**
```
tool: marketPerformance
endpoint: historical-sector-pe
sector: Technology
from_date: "2020-01-01"
```

---

## 19. `commodity`

Commodity market data: real-time quotes, historical EOD prices, and intraday charts for gold, oil, natural gas, wheat, copper, and more.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `commodities-list` | — | All available commodity symbols |
| `commodities-quote` | `symbol` | Real-time commodity quote |
| `commodities-quote-short` | `symbol` | Lightweight commodity quote |
| `all-commodities-quotes` | — | All commodity quotes at once |
| `commodities-historical-price-eod-full` | `symbol` | Full EOD price history |
| `commodities-historical-price-eod-light` | `symbol` | Light EOD price history |
| `commodities-intraday-1-min` | `symbol` | 1-minute intraday bars |
| `commodities-intraday-5-min` | `symbol` | 5-minute intraday bars |
| `commodities-intraday-1-hour` | `symbol` | 1-hour intraday bars |

**Parameters:** `symbol`, `from_date`, `to_date`, `short`

Common symbols: `GCUSD` (Gold), `CLUSD` (WTI Crude Oil), `NGUSD` (Natural Gas), `SIUSD` (Silver), `HGUSD` (Copper)

---

## 20. `crypto`

Cryptocurrency market data: real-time quotes, historical EOD prices, and intraday charts for Bitcoin, Ethereum, and thousands of other tokens.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `cryptocurrency-list` | — | All available crypto symbols |
| `cryptocurrency-quote` | `symbol` | Full real-time crypto quote |
| `cryptocurrency-quote-short` | `symbol` | Lightweight crypto quote |
| `all-cryptocurrency-quotes` | — | All crypto quotes at once |
| `cryptocurrency-historical-price-eod-full` | `symbol` | Full EOD price history |
| `cryptocurrency-historical-price-eod-light` | `symbol` | Light EOD price history |
| `cryptocurrency-intraday-1-min` | `symbol` | 1-minute intraday bars |
| `cryptocurrency-intraday-5-min` | `symbol` | 5-minute intraday bars |
| `cryptocurrency-intraday-1-hour` | `symbol` | 1-hour intraday bars |

**Parameters:** `symbol`, `from_date`, `to_date`, `short`

Common symbols: `BTCUSD`, `ETHUSD`, `SOLUSD`, `XRPUSD`

---

## 21. `forex`

Forex currency pair data: real-time quotes, historical EOD prices, and intraday charts.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `forex-list` | — | All available currency pairs |
| `forex-quote` | `symbol` | Full forex pair quote |
| `forex-quote-short` | `symbol` | Lightweight forex pair quote |
| `all-forex-quotes` | — | All forex pair quotes at once |
| `forex-historical-price-eod-full` | `symbol` | Full EOD price history |
| `forex-historical-price-eod-light` | `symbol` | Light EOD price history |
| `forex-intraday-1-min` | `symbol` | 1-minute intraday bars |
| `forex-intraday-5-min` | `symbol` | 5-minute intraday bars |
| `forex-intraday-1-hour` | `symbol` | 1-hour intraday bars |

**Parameters:** `symbol`, `from_date`, `to_date`, `short`

Common symbols: `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`

---

## 22. `etfAndMutualFunds`

ETF and mutual fund analysis: holdings, sector and country weightings, asset exposure, fund information, and SEC disclosure filings.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `information` | `symbol` | Fund overview (AUM, expense ratio, benchmark, inception) |
| `holdings` | `symbol` | Full list of fund holdings with weights |
| `sector-weighting` | `symbol` | Holdings breakdown by sector |
| `country-weighting` | `symbol` | Holdings breakdown by country |
| `etf-asset-exposure` | `symbol` | Asset class exposure breakdown |
| `latest-disclosures` | `symbol` | Most recent SEC disclosure filing |
| `mutual-fund-disclosures` | `symbol`, `year`, `quarter` | Mutual fund holdings for a specific quarter |
| `disclosures-dates` | `symbol` | Available disclosure filing dates |
| `disclosures-name-search` | `name` | Search fund disclosures by name |

**Parameters:** `symbol`, `cik`, `name`, `year`, `quarter`

---

## 23. `economics`

Macroeconomic indicators and data releases: GDP, CPI, unemployment, interest rates, treasury yields, and market risk premiums.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `economics-calendar` | — | Upcoming economic data release schedule |
| `economics-indicators` | `name` | Historical time series for a named indicator |
| `treasury-rates` | — | Current U.S. Treasury yield curve |
| `market-risk-premium` | — | Market risk premiums by country |

**Parameters:** `name`, `country`, `from_date`, `to_date`

**Common indicator names:** `GDP`, `realGDP`, `nominalPotentialGDP`, `realGDPPerCapita`, `federalFunds`, `CPI`, `inflationRate`, `inflation`, `retailSales`, `consumerSentiment`, `durableGoods`, `unemploymentRate`, `totalNonfarmPayroll`, `initialClaims`, `industrialProductionTotalIndex`, `newPrivatelyOwnedHousingUnitsStartedTotalUnits`, `totalVehicleSales`, `retailMoneyFunds`, `smoothedUSRecessionProbabilities`, `3MonthOr90DayRatesAndYieldsCertificatesOfDeposit`, `commercialBankCreditCard`, `commercialBankCreditCardInterestRateAsAPercentOfOutstandingBalances`, `nationalPopulation`

**Example:**
```
tool: economics
endpoint: economics-indicators
name: CPI
from_date: "2020-01-01"
```

---

## 24. `marketHours`

Exchange trading hours and holiday schedules for global exchanges.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `all-exchange-market-hours` | — | Hours and current open/closed status for all exchanges |
| `exchange-market-hours` | `exchange` | Hours for a specific exchange |
| `holidays-by-exchange` | `exchange` | Full holiday schedule for an exchange |

**Parameters:** `exchange`, `from_date`, `to_date`, `timestamp`

**Common exchange codes:** `NYSE`, `NASDAQ`, `LSE`, `TSX`, `ASX`, `HKEX`, `SSE`, `Euronext`, `JPX`

---

## 25. `Fundraisers`

SEC-filed crowdfunding campaigns and Regulation CF/A equity offerings. Search by company name or CIK, or browse the latest filings.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `latest-crowdfunding` | — | Most recent crowdfunding campaigns |
| `crowdfunding-search` | `name` | Search campaigns by company name |
| `crowdfunding-by-cik` | `cik` | Crowdfunding filings for a specific CIK |
| `latest-equity-offering` | — | Most recent Reg CF/A equity offerings |
| `equity-offering-search` | `name` | Search offerings by company name |
| `equity-offering-by-cik` | `cik` | Equity offerings for a specific CIK |

**Parameters:** `cik`, `name`, `limit`, `page`

---

## 26. `commitmentOfTraders`

*(Requires Premium, Ultimate, or Enterprise plan)*

CFTC Commitment of Traders (COT) reports showing positioning of commercial hedgers, large speculators, and small traders in futures markets.

**Endpoints:**

| Endpoint | Required params | Description |
|----------|----------------|-------------|
| `COT-report-list` | — | All symbols available in COT data |
| `COT-report` | `symbol` | Raw COT positioning data |
| `COT-report-analysis` | `symbol` | COT sentiment analysis and interpretation |

**Parameters:** `symbol`, `from_date`, `to_date`

**Example:**
```
tool: commitmentOfTraders
endpoint: COT-report-analysis
symbol: GC
from_date: "2026-01-01"
```

---

## 27. `directory`

Reference lists and symbol directories. Use these to discover valid symbols before querying other endpoints.

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `company-symbols-list` | All listed company stock symbols |
| `actively-trading-list` | All actively traded symbols |
| `ETFs-list` | All ETF symbols |
| `cik-list` | All company CIK numbers |
| `financial-symbols-list` | All symbols with financial statements available |
| `earnings-transcript-list` | All symbols with earnings transcripts available |
| `available-sectors` | All valid sector names |
| `available-industries` | All valid industry names |
| `available-exchanges` | All valid exchange codes |
| `available-countries` | All valid country codes |
| `symbol-changes-list` | History of ticker symbol changes (renames, mergers) |

**Parameters:** `extended` (boolean), `invalid` (boolean), `limit`, `page`

**Example:**
```
tool: directory
endpoint: available-sectors
```

---

## Common Patterns

### Comprehensive equity research data pull
```
company      → profile-symbol
statements   → income-statement, balance-sheet-statement, cashflow-statement, key-metrics-ttm, metrics-ratios-ttm
quote        → quote
analyst      → price-target-consensus, grades-summary
discountedCashFlow → dcf-advanced
insiderTrades → search-insider-trades
news         → search-stock-news
calendar     → earnings-company
company      → peers
```

### Peer valuation comparison
```
search       → search-company-screener (filter by sector)
quote        → batch-quote
statements   → key-metrics-ttm (per peer)
marketPerformance → sector-PE-snapshot
```

### Macro context overlay
```
economics    → economics-indicators (GDP, CPI, federalFunds)
economics    → treasury-rates
indexes      → sp-500, index-quote
marketPerformance → sector-performance-snapshot
```

### Institutional sentiment
```
form13F      → positions-summary
insiderTrades → insider-trade-statistics
analyst      → grades-summary, price-target-summary
senate       → senate-trading
```
