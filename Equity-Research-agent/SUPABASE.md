# Supabase Connection Guide

**Project:** RISKagent  
**Region:** eu-west-1  
**Dashboard:** https://supabase.com/dashboard/project/phmzdhbyliiqtwrltqmo

---

## Environment Variables

Add these to your `.env` file:

```env
SUPABASE_DB_URL=postgresql://postgres:YOUR_PASSWORD@db.phmzdhbyliiqtwrltqmo.supabase.co:5432/postgres
SUPABASE_URL=https://phmzdhbyliiqtwrltqmo.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
SUPABASE_SECRET_KEY=sb_secret_...
SUPABASE_S3_ENDPOINT=https://phmzdhbyliiqtwrltqmo.storage.supabase.co/storage/v1/s3
SUPABASE_S3_REGION=eu-west-1
SUPABASE_S3_ACCESS_KEY=...
SUPABASE_S3_SECRET_KEY=...
```

---

## Diagnosing Connection Issues from WSL

Run these steps in order. Each one narrows down where the failure is.

### 1. Check DNS resolution

```bash
getent hosts db.phmzdhbyliiqtwrltqmo.supabase.co
```

**Expected output:**
```
13.40.x.x    db.phmzdhbyliiqtwrltqmo.supabase.co
```

**If nothing is returned** — DNS is broken in WSL. Fix it:
```bash
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

Then re-run the `getent` command to confirm it resolves.

---

### 2. Check port reachability

```bash
nc -zv db.phmzdhbyliiqtwrltqmo.supabase.co 5432
```

**Expected output:**
```
Connection to db.phmzdhbyliiqtwrltqmo.supabase.co 5432 port [tcp/postgresql] succeeded!
```

**If you see `Network is unreachable` with an IPv6 address** — WSL resolved to IPv6 but has no IPv6 routing. Force IPv4:
```bash
nc -4 -zv db.phmzdhbyliiqtwrltqmo.supabase.co 5432
```

If `-4` succeeds, permanently prefer IPv4 in WSL:
```bash
echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf
```

Then restart the server — asyncpg will resolve to IPv4 and connect.

**If port 5432 is blocked entirely** — try the PgBouncer pooler on port 6543:
```bash
nc -zv db.phmzdhbyliiqtwrltqmo.supabase.co 6543
```

If port 6543 succeeds, update `SUPABASE_DB_URL` in `.env`:
```env
SUPABASE_DB_URL=postgresql://postgres:YOUR_PASSWORD@db.phmzdhbyliiqtwrltqmo.supabase.co:6543/postgres
```

---

### 3. Test the database connection with psql

Install the PostgreSQL client if not present:
```bash
sudo apt install -y postgresql-client
```

Connect:
```bash
psql "postgresql://postgres:YOUR_PASSWORD@db.phmzdhbyliiqtwrltqmo.supabase.co:5432/postgres?sslmode=require"
```

**If connected**, verify the table exists and check row count:
```sql
\dt agent_responses
select count(*) from agent_responses;
\q
```

---

### 4. Verify the server picked up the DB connection

When you run `bash start.sh`, look for this line in the terminal output:

```
[db] Supabase pool connected ✓
```

If you see this instead:
```
[db] Supabase unreachable at startup — saves disabled: ...
```

The pool failed. Fix DNS/port (steps 1–2 above) then restart the server.

---

## Querying Saved Responses

Open the **SQL Editor** in the Supabase dashboard and run:

```sql
-- Latest 10 responses across all agents
select id, agent_name, model, query, created_at, left(response, 120)
from agent_responses
order by created_at desc
limit 10;

-- Filter by agent
select * from agent_responses
where agent_name = 'equity-research'
order by created_at desc;

-- Filter by date
select * from agent_responses
where created_at::date = current_date
order by created_at desc;
```

---

## Storage Bucket

Supabase Storage connects via the S3-compatible endpoint over HTTPS. No SSL enforcement setting on the Postgres database is needed — storage and the database are independent services. Files are visible in the Supabase dashboard once the S3 credentials are correctly set in `.env`.

Agent responses are uploaded as markdown files to the `insuranceRISKagent` bucket:

| Path | Source |
|------|--------|
| `equity-research/{TICKER}-{date}.md` | `node agent.js` runs |
| `transactional-liability-wi-agent/{date}-transactional-liability-wi-agent.md` | Chat UI — W&I agent |
| `chief-capital-modelling-agent/{date}-chief-capital-modelling-agent.md` | Chat UI — Capital Modelling agent |
| `sec-filings-analyst/{date}-sec-filings-analyst.md` | Chat UI — SEC Filings Analyst agent |

View files: **Supabase dashboard → Storage → insuranceRISKagent**

---

## Persistence — How Saves Work

Agent responses are written through two layers, in order:

**1. Direct Postgres pool (asyncpg → `agent_responses` table)**
The preferred path. The pool is established at server startup with `ssl="require"`. If the pool is healthy, inserts go directly to Postgres. Look for `[db] Supabase pool connected ✓` in the server log to confirm.

**2. REST API fallback (httpx → `/rest/v1/agent_responses`)**
If the direct pool is unavailable at startup (DNS failure, port blocked, etc.), the server automatically falls back to the Supabase REST API via `httpx`. Saves still land in the same `agent_responses` table — they just go through the PostgREST layer instead of a raw Postgres connection. Look for `[db] Supabase REST save ✓` in logs.

**3. Supabase Storage (S3-compatible via boto3 / @aws-sdk/client-s3)**
Runs in parallel with the database save. Uploads the response as a markdown file to the `insuranceRISKagent` bucket using Supabase's S3-compatible endpoint. Look for `[supabase-s3] uploaded →` in logs.

**4. AWS S3 (primary cloud storage bucket)**
Runs in parallel with the Supabase upload — both are attempted on every save regardless of each other's outcome. Uploads the same markdown file to the bucket named in `AWS_S3_BUCKET`. Uses standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_REGION`). Look for `[aws-s3] uploaded →` in logs. Silently skipped if credentials or bucket name are not set.

**5. Local markdown file**
Always written to `sample-outputs/` as a dated `.md` file. This is the zero-dependency fallback — it works even if all remote connections are down.

---

## Table Schema

```sql
create table agent_responses (
  id          uuid primary key default gen_random_uuid(),
  agent_name  text not null,
  model       text,
  query       text not null,
  response    text not null,
  created_at  timestamptz default now()
);
```
