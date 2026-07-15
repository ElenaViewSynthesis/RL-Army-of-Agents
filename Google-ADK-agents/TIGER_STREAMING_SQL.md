# Tiger Cloud (TimescaleDB) — streaming SQL for `commodity_prices`

SQL reference for the commodity price time-series on **Tiger Cloud**
(TimescaleDB). This is the *numbers* half of the system — Supabase keeps the
*words* (`agent_responses` + pgvector). Written/read from `a2a_finance/tiger_client.py`
and `seed_timescale_prices.py`.

| Field | Value |
|-------|-------|
| Engine | PostgreSQL 18.4 + TimescaleDB 2.28.2 |
| Host / port | `i2t2hp8zb1.v445e4qjbc.tsdb.cloud.timescale.com` : `32445` |
| Database | `tsdb` (user `tsdbadmin`) |
| Connection | `TIGER_DATABASE_*` in gitignored `finance_coordinator/.env` (`sslmode=require`) |

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

## 4. Streaming / time-series patterns (recommended — not yet applied)

TimescaleDB-native queries for the dense (~hourly) OilPrice series.

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

> Section 4 is a menu, not applied state. Create the continuous aggregate only
> once enough raw data has accumulated; the retention policy is optional.
