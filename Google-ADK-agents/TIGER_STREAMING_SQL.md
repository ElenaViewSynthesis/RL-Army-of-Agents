# Tiger Cloud (TimescaleDB) — streaming SQL for `commodity_prices`

SQL reference for the Tiger Cloud (TimescaleDB) time-series — the *numbers* half
of the system, while Supabase keeps the *words* (`agent_responses` + pgvector).
Two hypertables: **`commodity_prices`** (§1–4) and **`insider_trades`** (§5).
Written/read from `a2a_finance/tiger_client.py` via `seed_timescale_prices.py`
and `seed_insider_trades.py`.

| Field | Value |
|-------|-------|
| Engine | PostgreSQL 18.4 + TimescaleDB 2.28.2 |
| Hosting | AWS — managed by **Tiger Cloud** |
| Compute | 8 CPUs (service allocation) |
| Host / port | `i2t2hp8zb1.v445e4qjbc.tsdb.cloud.timescale.com` : `32445` |
| Database | `tsdb` (user `tsdbadmin`) |
| Connection | `TIGER_DATABASE_*` in gitignored `finance_coordinator/.env` (`sslmode=require`) |

**Stack roles:**
- **Tiger Cloud** hosts the managed database on **AWS**.
- **TimescaleDB** is the PostgreSQL **extension** that adds hypertables and
  continuous aggregates.
- The **hypertable partitions `commodity_prices`** by time (`ts`).
- **8 CPUs** is the service's compute allocation (server-side; no client change).

Codes are normalized (`WTI`, `BRENT`, `NATGAS`, …); `source` is `oilprice` | `fmp`.

---

## 1. Schema — hypertable (executed)

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS commodity_prices (
    ts            timestamptz      NOT NULL,   -- time dimension (hypertable partition)
    code          text             NOT NULL,   -- normalized: WTI / BRENT / NATGAS / …
    source        text             NOT NULL,   -- 'oilprice' | 'fmp'
    source_symbol text,                         -- native symbol: WTI_USD / CLUSD
    name          text,
    price         double precision,
    currency      text,
    unit          text,
    change        double precision,
    PRIMARY KEY (code, source, ts)              -- includes ts (required by hypertables)
);

-- Turn it into a hypertable partitioned on ts.
SELECT create_hypertable('commodity_prices', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS commodity_prices_code_ts
    ON commodity_prices (code, ts DESC);
```

> The PK `(code, source, ts)` must include the partitioning column `ts` — a
> TimescaleDB requirement — and is exactly what makes the upsert idempotent.

---

## 2. Idempotent ingest — upsert (executed each seed run)

`tiger_client.save_prices()` upserts on the PK, so overlapping day windows are
safe to re-run (a missed run self-heals; re-runs never duplicate):

```sql
INSERT INTO commodity_prices
    (ts, code, source, source_symbol, name, price, currency, unit, change)
VALUES
    (%(ts)s, %(code)s, %(source)s, %(source_symbol)s, %(name)s,
     %(price)s, %(currency)s, %(unit)s, %(change)s)
ON CONFLICT (code, source, ts) DO UPDATE SET
    price         = EXCLUDED.price,
    currency      = EXCLUDED.currency,
    unit          = EXCLUDED.unit,
    change        = EXCLUDED.change,
    source_symbol = EXCLUDED.source_symbol,
    name          = EXCLUDED.name;
```

---

## 3. Verification / inspection (executed)

```sql
-- Coverage per code + source: how many points, date span, price range.
SELECT code, source, count(*),
       min(ts)::date, max(ts)::date,
       round(min(price)::numeric, 2), round(max(price)::numeric, 2)
FROM commodity_prices
GROUP BY code, source
ORDER BY code, source;

-- Most recent points across the table.
SELECT code, source, ts, price, currency, unit
FROM commodity_prices
ORDER BY ts DESC
LIMIT 10;

-- Hypertable metadata.
SELECT hypertable_name, num_dimensions
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'commodity_prices';
```

---

## 4. Streaming / time-series patterns

TimescaleDB-native queries for the dense (~hourly) OilPrice series. The ad-hoc
queries below are examples; the **continuous aggregate + refresh policy are
applied** (see the note under §4's CAGG block).

```sql
-- Latest price per code (one row each), newest first.
SELECT DISTINCT ON (code) code, source, ts, price, currency
FROM commodity_prices
ORDER BY code, ts DESC;
```

```sql
-- Hourly OHLC rollup with time_bucket + first()/last() (Timescale aggregates).
SELECT code,
       time_bucket('1 hour', ts) AS bucket,
       first(price, ts) AS open,
       max(price)       AS high,
       min(price)       AS low,
       last(price, ts)  AS close
FROM commodity_prices
WHERE source = 'oilprice'
GROUP BY code, bucket
ORDER BY code, bucket DESC;
```

```sql
-- Continuous aggregate: a "streaming" materialized daily OHLC that stays fresh
-- as new points arrive (incremental refresh, not a full recompute).
CREATE MATERIALIZED VIEW commodity_prices_daily
WITH (timescaledb.continuous) AS
SELECT code, source,
       time_bucket('1 day', ts) AS day,
       first(price, ts) AS open,
       max(price)       AS high,
       min(price)       AS low,
       last(price, ts)  AS close,
       count(*)         AS n
FROM commodity_prices
GROUP BY code, source, day
WITH NO DATA;

-- Refresh policy — keeps the aggregate streaming (hourly, over the last 3 days).
SELECT add_continuous_aggregate_policy('commodity_prices_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

```sql
-- Optional retention: drop raw points older than 90 days (aggregate persists).
SELECT add_retention_policy('commodity_prices', INTERVAL '90 days');
```

> **Applied:** `commodity_prices_daily` (continuous aggregate) + its hourly
> refresh policy exist on the service (created after the first full seed). Query
> it directly — real-time aggregation unions materialized buckets with the
> latest raw points. The **retention policy above is NOT applied** (optional;
> add it if/when raw-data volume warrants).

---

## 5. Insider trades — second hypertable + daily aggregate

A parallel dataset built the same way: FMP's free `insider-trading/latest` feed
accumulated into the `insider_trades` hypertable (only **important** trades —
Form 4 open-market buys/sells ≥ $50k). Written by
`a2a_finance/tiger_client.save_insider_trades()` via `seed_insider_trades.py`.

### Schema (executed)

```sql
CREATE TABLE IF NOT EXISTS insider_trades (
    transaction_date        timestamptz NOT NULL,   -- time dimension (partition)
    trade_id                text        NOT NULL,   -- dedup hash (client-computed)
    symbol                  text        NOT NULL,
    company_cik text, reporting_cik text, reporting_name text, type_of_owner text,
    transaction_type text, acquisition_disposition text,   -- A = buy, D = sell
    securities_transacted double precision, price double precision,
    value double precision, securities_owned double precision,
    form_type text, security_name text, filing_date date, url text,
    PRIMARY KEY (transaction_date, trade_id)
);

SELECT create_hypertable('insider_trades', 'transaction_date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS insider_trades_symbol_ts
    ON insider_trades (symbol, transaction_date DESC);
```

### Idempotent ingest — upsert (executed each poll)

```sql
INSERT INTO insider_trades
    (trade_id, transaction_date, symbol, company_cik, reporting_cik, reporting_name,
     type_of_owner, transaction_type, acquisition_disposition, securities_transacted,
     price, value, securities_owned, form_type, security_name, filing_date, url)
VALUES (%(trade_id)s, %(transaction_date)s, …)
ON CONFLICT (transaction_date, trade_id) DO UPDATE SET
    securities_owned = EXCLUDED.securities_owned,
    url              = EXCLUDED.url;
```

### Daily buy/sell continuous aggregate (executed)

Grouped by `acquisition_disposition` (A = buy, D = sell) — avoids `FILTER`/`CASE`,
which continuous aggregates restrict:

```sql
CREATE MATERIALIZED VIEW insider_trades_daily
WITH (timescaledb.continuous) AS
SELECT symbol, acquisition_disposition,
       time_bucket('1 day', transaction_date) AS day,
       count(*)                   AS n_trades,
       sum(value)                 AS total_value,
       sum(securities_transacted) AS total_shares
FROM insider_trades
GROUP BY symbol, acquisition_disposition, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy('insider_trades_daily',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

### Query patterns

```sql
-- Daily net insider $ flow per symbol (buy vs sell), from the aggregate.
SELECT symbol, day,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'A') AS buy_usd,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'D') AS sell_usd
FROM insider_trades_daily
GROUP BY symbol, day
ORDER BY day DESC;

-- Biggest single insider trades stored (raw table).
SELECT symbol, transaction_type, securities_transacted, price,
       round(value::numeric, 0) AS value, type_of_owner
FROM insider_trades
ORDER BY value DESC
LIMIT 20;
```

> **Applied:** the `insider_trades` hypertable + `insider_trades_daily` continuous
> aggregate and its hourly refresh policy exist on the service. Retention on the
> raw table is optional/unapplied (see §4).

---

## 6. TLS-encrypted database connection

TLS is not a database type. It encrypts network traffic between this application
and Tiger Cloud so database credentials, SQL queries, and returned data are not
sent as plaintext. The connection URL enables TLS with `sslmode=require`:

```dotenv
TIGER_DATABASE_URL=postgresql://<user>:<url-encoded-password>@<host>:5432/tsdb?sslmode=require
```

When `psycopg` reads this URL, it performs the following connection sequence:

1. Open a TCP connection to the Tiger Cloud PostgreSQL endpoint.
2. Request a PostgreSQL TLS connection before sending authentication details.
3. Receive the server certificate and negotiate TLS encryption keys.
4. Send authentication data, SQL queries, and query results through the
   encrypted channel.
5. Fail the connection if TLS cannot be established. `require` does not permit
   a fallback to an unencrypted connection.

The available PostgreSQL SSL modes provide different levels of protection:

| Mode | Behavior |
|---|---|
| `disable` | Do not use TLS. |
| `prefer` | Try TLS first, but permit an unencrypted fallback. |
| `require` | Require an encrypted TLS connection. |
| `verify-ca` | Require TLS and verify that a trusted certificate authority issued the server certificate. |
| `verify-full` | Require TLS, verify the certificate authority, and verify that the certificate matches the database hostname. |

`sslmode=require` guarantees encryption and prevents a plaintext downgrade, but
the setting alone does not require full hostname verification. Use
`sslmode=verify-full` with the appropriate CA certificate when strict server
identity verification is required. TLS protects data in transit; encryption at
rest is a separate database and storage configuration.
