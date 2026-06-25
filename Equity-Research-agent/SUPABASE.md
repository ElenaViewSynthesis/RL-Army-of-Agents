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

**If it times out or is refused** — port 5432 is blocked. Try the PgBouncer pooler on port 6543:
```bash
nc -zv db.phmzdhbyliiqtwrltqmo.supabase.co 6543
```

If port 6543 succeeds, update `SUPABASE_DB_URL` in `.env` to use port 6543:
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

Agent responses are also uploaded as markdown files to the `insuranceRISKagent` bucket.

| Path | Source |
|------|--------|
| `equity-research/{TICKER}-{date}.md` | `node agent.js` runs |
| `transactional-liability-wi-agent/{date}-transactional-liability-wi-agent.md` | Chat UI — W&I agent |
| `chief-capital-modelling-agent/{date}-chief-capital-modelling-agent.md` | Chat UI — Capital Modelling agent |

View files: **Supabase dashboard → Storage → insuranceRISKagent**

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
