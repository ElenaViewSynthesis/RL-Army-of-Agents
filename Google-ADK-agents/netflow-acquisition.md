# Net insider flow — acquisition vs disposition

SQL for reading **net insider dollar flow** out of the TimescaleDB
`insider_trades` hypertable and its `insider_trades_daily` continuous aggregate.
`acquisition_disposition` is `A` (acquired = buy) or `D` (disposed = sell); the
aggregate groups on it, so you **pivot buy vs sell at query time** with `FILTER`
(FILTER isn't allowed *inside* a continuous aggregate, but is fine when reading).

Connection: `TIGER_DATABASE_*` in gitignored `finance_coordinator/.env`
(`sslmode=require`). See [`TIGER_STREAMING_SQL.md`](TIGER_STREAMING_SQL.md) for
the schema and aggregate DDL.

---

## Net buy/sell $ per symbol per day (from the aggregate)

```sql
SELECT symbol, day,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'A') AS buy_usd,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'D') AS sell_usd,
       sum(n_trades)                                                 AS n
FROM insider_trades_daily
GROUP BY symbol, day
ORDER BY day DESC, sell_usd DESC NULLS LAST
LIMIT 10;
```

## Net flow (buy − sell) per symbol over a window

```sql
SELECT symbol,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'A') AS buy_usd,
       sum(total_value) FILTER (WHERE acquisition_disposition = 'D') AS sell_usd,
       coalesce(sum(total_value) FILTER (WHERE acquisition_disposition = 'A'), 0)
     - coalesce(sum(total_value) FILTER (WHERE acquisition_disposition = 'D'), 0)
         AS net_usd
FROM insider_trades_daily
WHERE day >= now() - interval '30 days'
GROUP BY symbol
ORDER BY net_usd;          -- most net selling first; DESC for most net buying
```

## Sanity checks

```sql
-- Aggregate row count (materialized + real-time rows).
SELECT count(*) FROM insider_trades_daily;

-- Raw rows by direction (A = buy, D = sell).
SELECT acquisition_disposition, count(*)
FROM insider_trades
GROUP BY acquisition_disposition;
```

## Biggest single trades (raw table)

```sql
SELECT symbol, transaction_type, securities_transacted, price,
       round(value::numeric, 0) AS value, type_of_owner
FROM insider_trades
ORDER BY value DESC
LIMIT 20;
```

> The aggregate refreshes hourly (server-side policy), and real-time aggregation
> unions materialized buckets with the latest raw rows — so newly polled trades
> show up in these queries immediately.
