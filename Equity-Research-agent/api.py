"""
Equity Research FastAPI — wraps the Node.js agent and indices scripts,
exposing them as REST endpoints with optional SSE streaming.

Start: uvicorn api:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import asyncpg
import boto3
import httpx
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR   = Path(__file__).parent
AGENTS_DIR = BASE_DIR / "agents"
FMP_STABLE = "https://financialmodelingprep.com/stable"
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Equity Research API",
    description="AI-powered institutional equity research via Nemotron + FMP",
    version="1.0.0",
)

MODEL_ALIASES = {
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "laguna":   "poolside/laguna-m.1:free",
}

FINANCE_SYSTEM_PROMPT = """You are a specialist financial analyst focused on emerging markets ETFs and equities. Your expertise covers developing economy stocks and funds across Asia, Latin America, Eastern Europe, the Middle East, and Africa.

Guidelines:
- Ticker symbols always refer to publicly traded securities — interpret them in a financial context without asking for clarification.
- For emerging market ETFs (e.g. VWO, EEM, SPYG): cover the full name, index tracked, expense ratio, AUM, top country and sector weights, top holdings, liquidity (average daily volume), and the investment case vs. developed-market alternatives.
- For emerging market stocks (e.g. BABA, NIO, BIDU, TSM): cover business model, home market dynamics, ADR structure if applicable, key financial metrics, geopolitical risk, and currency exposure.
- Always contextualise against macro factors relevant to EM: USD strength, commodity cycles, China policy risk, Fed rate trajectory, and EM capital flows.
- Use precise figures and percentages. Be direct and opinionated — avoid vague generalisations."""

# ── database pool ─────────────────────────────────────────────────────────────

_db_pool: asyncpg.Pool | None = None


@app.on_event("startup")
async def _startup():
    global _db_pool
    db_url = os.environ.get("SUPABASE_DB_URL")
    if db_url:
        try:
            _db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5, ssl="require")
            print("[db] Supabase pool connected ✓")
        except Exception as e:
            print(f"[db] Supabase unreachable at startup — saves disabled: {e}")


@app.on_event("shutdown")
async def _shutdown():
    if _db_pool:
        await _db_pool.close()


async def _save_agent_response(agent_name: str, model: str, query: str, response: str):
    """Persist agent response to Supabase and write a markdown file to sample-outputs/."""
    full_model = MODEL_ALIASES.get(model, model)
    today = date.today().isoformat()
    header = (
        f"# Agent Response — {agent_name}\n\n"
        f"**Query:** {query}\n\n"
        f"**Agent:** {agent_name}\n"
        f"**Model:** {full_model}\n"
        f"**Date:** {today}\n\n"
        f"---\n\n"
    )
    payload = {
        "agent_name": agent_name,
        "model": full_model,
        "query": query,
        "response": response,
    }

    # ── Supabase insert ───────────────────────────────────────────────────────
    saved = False
    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO agent_responses (agent_name, model, query, response)
                       VALUES ($1, $2, $3, $4)""",
                    agent_name, full_model, query, response,
                )
            saved = True
        except Exception as e:
            print(f"[db] save failed: {e}")

    if not saved:
        await _save_agent_response_rest(payload)

    # ── Supabase Storage (S3) ─────────────────────────────────────────────────
    _upload_to_storage(f"{agent_name}/{today}-{agent_name}.md", header + response)

    # ── markdown file ─────────────────────────────────────────────────────────
    out_dir = BASE_DIR / "sample-outputs"
    out_dir.mkdir(exist_ok=True)
    base = f"{agent_name}-{today}"
    path = out_dir / f"{base}.md"
    counter = 1
    while path.exists():
        path = out_dir / f"{base}-{counter}.md"
        counter += 1
    path.write_text(header + response + "\n", encoding="utf-8")
    print(f"[save] {path.name}")


async def _save_agent_response_rest(payload: dict):
    """Persist via Supabase REST when the direct Postgres route is unavailable."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not (url and key):
        print("[db] Supabase REST credentials not set — save skipped")
        return

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/rest/v1/agent_responses",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=payload,
            )
            resp.raise_for_status()
        print("[db] Supabase REST save ✓")
    except Exception as e:
        print(f"[db] Supabase REST save failed: {e}")


def _upload_to_storage(key: str, body: str):
    """Upload a text file to Supabase Storage via S3 protocol (best-effort)."""
    endpoint   = os.environ.get("SUPABASE_S3_ENDPOINT")
    access_key = os.environ.get("SUPABASE_S3_ACCESS_KEY")
    secret_key = os.environ.get("SUPABASE_S3_SECRET_KEY")
    region     = os.environ.get("SUPABASE_S3_REGION", "eu-west-1")
    if not (endpoint and access_key and secret_key):
        return
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}),
        )
        s3.put_object(
            Bucket="insuranceRISKagent",
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )
        print(f"[storage] uploaded → insuranceRISKagent/{key}")
    except Exception as e:
        print(f"[storage] upload failed: {e}")


# ── internal helpers ──────────────────────────────────────────────────────────

def _agent_args(ticker: str, model: str, save: bool) -> list[str]:
    args = ["agent.js", ticker.upper(), f"--model={model}"]
    if save:
        args.append("--save")
    return args


async def _run(script_args: list[str], timeout: int = 360) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        "node", *script_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(BASE_DIR),
        env={**os.environ},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(), stderr.decode(), proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail=f"Agent timed out after {timeout}s")


async def _stream_sse(script_args: list[str]):
    """
    Async generator yielding Server-Sent Events:
      - progress events from stderr (step-by-step tool calls, reasoning tokens)
      - report event from stdout (final markdown report)
      - done event on completion
    """
    proc = await asyncio.create_subprocess_exec(
        "node", *script_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(BASE_DIR),
        env={**os.environ},
    )

    # Stream stderr line-by-line as progress events
    async for raw in proc.stderr:
        line = raw.decode().rstrip()
        if line:
            yield f"data: {json.dumps({'type': 'progress', 'message': line})}\n\n"
            await asyncio.sleep(0)

    # Process is done — read full report from stdout
    stdout = await proc.stdout.read()
    if stdout:
        yield f"data: {json.dumps({'type': 'report', 'content': stdout.decode()})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── symbols cache ─────────────────────────────────────────────────────────────

_FREE_TIER_SYMBOLS = [
    {"symbol":"AAPL",  "name":"Apple Inc.",                       "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"ABBV",  "name":"AbbVie Inc.",                      "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"ADBE",  "name":"Adobe Inc.",                       "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"AAL",   "name":"American Airlines Group",          "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"AMD",   "name":"Advanced Micro Devices",           "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"AMZN",  "name":"Amazon.com Inc.",                  "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"ATVI",  "name":"Activision Blizzard",              "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"BA",    "name":"Boeing Co.",                       "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"BAC",   "name":"Bank of America Corp.",            "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"BABA",  "name":"Alibaba Group Holding",            "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"BIDU",  "name":"Baidu Inc.",                       "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"BILI",  "name":"Bilibili Inc.",                    "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"C",     "name":"Citigroup Inc.",                   "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"CARR",  "name":"Carrier Global Corp.",             "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"CCL",   "name":"Carnival Corporation",             "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"COIN",  "name":"Coinbase Global Inc.",             "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"COST",  "name":"Costco Wholesale Corp.",           "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"CPRX",  "name":"Catalyst Pharmaceuticals",         "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"CSCO",  "name":"Cisco Systems Inc.",               "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"CVX",   "name":"Chevron Corporation",              "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"DAL",   "name":"Delta Air Lines Inc.",             "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"DIS",   "name":"Walt Disney Co.",                  "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"DOCU",  "name":"DocuSign Inc.",                    "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"ET",    "name":"Energy Transfer LP",               "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"ETSY",  "name":"Etsy Inc.",                        "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"F",     "name":"Ford Motor Co.",                   "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"FDX",   "name":"FedEx Corporation",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"FUBO",  "name":"fuboTV Inc.",                      "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"GE",    "name":"GE Aerospace",                     "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"GM",    "name":"General Motors Co.",               "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"GOOGL", "name":"Alphabet Inc.",                    "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"GS",    "name":"Goldman Sachs Group",              "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"HCA",   "name":"HCA Healthcare Inc.",              "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"HOOD",  "name":"Robinhood Markets Inc.",           "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"INTC",  "name":"Intel Corporation",                "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"JNJ",   "name":"Johnson & Johnson",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"JPM",   "name":"JPMorgan Chase & Co.",             "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"KO",    "name":"Coca-Cola Co.",                    "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"LCID",  "name":"Lucid Group Inc.",                 "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"LMT",   "name":"Lockheed Martin Corp.",            "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"META",  "name":"Meta Platforms Inc.",              "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"MGM",   "name":"MGM Resorts International",        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"MRNA",  "name":"Moderna Inc.",                     "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"MRO",   "name":"Marathon Oil Corporation",         "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"MSFT",  "name":"Microsoft Corporation",            "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"NFLX",  "name":"Netflix Inc.",                     "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"NIO",   "name":"NIO Inc.",                         "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"NKE",   "name":"Nike Inc.",                        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"NOK",   "name":"Nokia Corporation",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"NVDA",  "name":"NVIDIA Corporation",               "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"PEP",   "name":"PepsiCo Inc.",                     "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"PFE",   "name":"Pfizer Inc.",                      "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"PINS",  "name":"Pinterest Inc.",                   "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"PLTR",  "name":"Palantir Technologies",            "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"PYPL",  "name":"PayPal Holdings Inc.",             "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"RBLX",  "name":"Roblox Corporation",               "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"RIOT",  "name":"Riot Platforms Inc.",              "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"RIVN",  "name":"Rivian Automotive Inc.",           "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"RKT",   "name":"Rocket Companies Inc.",            "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"ROKU",  "name":"Roku Inc.",                        "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"SBUX",  "name":"Starbucks Corporation",            "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"SHOP",  "name":"Shopify Inc.",                     "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"SIRI",  "name":"Sirius XM Holdings",               "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"SNAP",  "name":"Snap Inc.",                        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"SOFI",  "name":"SoFi Technologies",                "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"SONY",  "name":"Sony Group Corporation",           "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"SPY",   "name":"SPDR S&P 500 ETF Trust",           "exchangeShortName":"NYSE",  "type":"etf"},
    {"symbol":"SPYG",  "name":"SPDR Portfolio S&P 500 Growth ETF","exchangeShortName":"NYSE",  "type":"etf"},
    {"symbol":"SQ",    "name":"Block Inc.",                       "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"T",     "name":"AT&T Inc.",                        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"TGT",   "name":"Target Corporation",               "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"TLRY",  "name":"Tilray Brands Inc.",               "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"TSM",   "name":"Taiwan Semiconductor Mfg.",        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"TSLA",  "name":"Tesla Inc.",                       "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"TWTR",  "name":"Twitter / X Corp.",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"UAL",   "name":"United Airlines Holdings",         "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"UBER",  "name":"Uber Technologies Inc.",           "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"UNH",   "name":"UnitedHealth Group Inc.",          "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"V",     "name":"Visa Inc.",                        "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"VIAC",  "name":"Paramount Global (ex-ViacomCBS)",  "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"VWO",   "name":"Vanguard FTSE Emerging Markets ETF","exchangeShortName":"NYSE", "type":"etf"},
    {"symbol":"VZ",    "name":"Verizon Communications",           "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"WBA",   "name":"Walgreens Boots Alliance",         "exchangeShortName":"NASDAQ","type":"stock"},
    {"symbol":"WFC",   "name":"Wells Fargo & Co.",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"WMT",   "name":"Walmart Inc.",                     "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"XOM",   "name":"Exxon Mobil Corp.",                "exchangeShortName":"NYSE",  "type":"stock"},
    {"symbol":"ZM",    "name":"Zoom Video Communications",        "exchangeShortName":"NASDAQ","type":"stock"},
]

_symbols_cache: dict = {"data": None, "ts": 0.0}


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/symbols")
async def list_symbols():
    """Return the hardcoded list of FMP free-tier supported symbols."""
    return {"symbols": _FREE_TIER_SYMBOLS, "count": len(_FREE_TIER_SYMBOLS)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_model": "nemotron",
        "available_models": MODEL_ALIASES,
    }


class ResearchResult(BaseModel):
    ticker: str
    model: str
    report: str
    duration_seconds: float


@app.post("/research/{ticker}", response_model=ResearchResult)
async def research(
    ticker: str,
    model: str = Query(default="nemotron", description="nemotron | laguna | any OpenRouter model ID"),
    save: bool  = Query(default=True,      description="Write report to output/"),
):
    """
    Run a full equity research report and return it as JSON.
    Blocks until the agent finishes (typically 1-3 minutes).
    For real-time progress use GET /research/{ticker}/stream instead.
    """
    t0 = time.time()
    stdout, stderr, code = await _run(_agent_args(ticker, model, save))

    if code != 0:
        err = next((l for l in stderr.splitlines() if "Error:" in l), stderr[-400:])
        raise HTTPException(status_code=500, detail=err)

    return ResearchResult(
        ticker=ticker.upper(),
        model=model,
        report=stdout,
        duration_seconds=round(time.time() - t0, 1),
    )


@app.get("/research/{ticker}/stream")
async def research_stream(
    ticker: str,
    model: str = Query(default="nemotron"),
    save: bool  = Query(default=True),
):
    """
    Stream the agent's progress and final report as Server-Sent Events.

    Event types:
      { "type": "progress", "message": "..." }   — step-by-step tool calls
      { "type": "report",   "content": "..." }   — final markdown report
      { "type": "done" }                          — stream complete
    """
    return StreamingResponse(
        _stream_sse(_agent_args(ticker, model, save)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/indices")
async def indices():
    """
    Run the global market indices snapshot (direct FMP fetch, no LLM).
    Returns structured markdown covering VIX regime, daily performance,
    breadth, 52-week positioning, MA signals, and regional summary.
    """
    t0 = time.time()
    stdout, stderr, code = await _run(["indices.js"])
    if code != 0:
        raise HTTPException(status_code=500, detail=stderr[-400:])
    return {"report": stdout, "duration_seconds": round(time.time() - t0, 1)}


@app.get("/output")
async def list_reports():
    """List all saved reports in output/, newest first."""
    output_dir = BASE_DIR / "output"
    if not output_dir.exists():
        return {"files": []}
    files = sorted(output_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {"files": [f.name for f in files]}


@app.get("/output/{filename}")
async def get_report(filename: str):
    """Fetch a saved report by filename."""
    path = BASE_DIR / "output" / filename
    if not path.exists() or path.suffix != ".md":
        raise HTTPException(status_code=404, detail="Report not found")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}


# ── chat completions ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model: str = "nemotron"
    system: str | None = None


# ── agent definitions ─────────────────────────────────────────────────────────

@app.get("/agents")
async def list_agents():
    """List all available agent definitions in agents/."""
    if not AGENTS_DIR.exists():
        return {"agents": []}
    agents = [
        {"name": f.stem, "filename": f.name}
        for f in sorted(AGENTS_DIR.glob("*.md"))
    ]
    return {"agents": agents}


@app.post("/agents/{name}/chat/stream")
async def agent_chat_stream(name: str, req: ChatRequest):
    """
    Chat with a named agent using its .md definition as the system prompt.
    Streams SSE tokens identical to /chat/stream.

    Available agents: chief-capital-modelling-agent, transactional-liability-wi-agent
    """
    path = AGENTS_DIR / f"{name}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found. GET /agents for the list.")
    system_prompt = path.read_text(encoding="utf-8")

    user_content = req.message
    if name == "sec-filings-analyst":
        context = await _fetch_sec_filings_context()
        if context:
            user_content = f"{req.message}\n\n---\n\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return StreamingResponse(
        _agent_stream_and_save(name, messages, req.model, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _agent_stream_and_save(agent_name: str, messages: list, model: str, query: str):
    """Wraps _openrouter_stream: passes tokens through and saves the full response on completion."""
    accumulated: list[str] = []
    try:
        async for chunk in _openrouter_stream(messages, model):
            yield chunk
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:].strip())
                    if "content" in data:
                        accumulated.append(data["content"])
                except (json.JSONDecodeError, KeyError):
                    pass
    finally:
        if accumulated:
            asyncio.create_task(
                _save_agent_response(agent_name, model, query, "".join(accumulated))
            )


async def _openrouter_stream(messages: list, model: str):
    full_model = MODEL_ALIASES.get(model, model)
    # connect_timeout=10s; read_timeout=660s (11 min) — free-tier queue can take 7-10 min
    timeout = httpx.Timeout(connect=10.0, read=660.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={"model": full_model, "messages": messages, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"data: {json.dumps({'error': f'OpenRouter {resp.status_code}: {body.decode()[:200]}'})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        # surface OpenRouter error objects (e.g. rate limit)
                        if "error" in chunk:
                            yield f"data: {json.dumps({'error': chunk['error'].get('message', str(chunk['error']))})}\n\n"
                            return
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except (KeyError, json.JSONDecodeError):
                        continue
    except httpx.ReadTimeout:
        yield f"data: {json.dumps({'error': 'Request timed out after 11 minutes — OpenRouter queue may be overloaded. Try again or add credits at openrouter.ai/credits.'})}\n\n"
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Cannot reach OpenRouter — check your internet connection.'})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.get("/sec-filings")
async def sec_filings(
    from_date: str = Query(default=None, alias="from", description="Start date YYYY-MM-DD (default: 30 days ago)"),
    to_date:   str = Query(default=None, alias="to",   description="End date YYYY-MM-DD (default: today)"),
    page:      int = Query(default=0,    description="Page index"),
    limit:     int = Query(default=20,   description="Results per page (max 100)"),
):
    """
    Fetch recent 8-K SEC filings from FMP (premium endpoint).
    Returns raw FMP response. Requires a paid FMP API key.
    """
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY not configured")

    today = date.today()
    params = {
        "from":   from_date or (today - timedelta(days=30)).isoformat(),
        "to":     to_date   or today.isoformat(),
        "page":   page,
        "limit":  min(limit, 100),
        "apikey": fmp_key,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{FMP_STABLE}/sec-filings-8k", params=params)
            resp.raise_for_status()
            filings = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"FMP error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMP request failed: {e}")

    return {"filings": filings, "count": len(filings), "params": params}


async def _fetch_sec_filings_context(days: int = 30, limit: int = 50) -> str:
    """Fetch recent 8-K filings and format them as a compact context block for the LLM."""
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        return ""
    today = date.today()
    params = {
        "from":   (today - timedelta(days=days)).isoformat(),
        "to":     today.isoformat(),
        "page":   0,
        "limit":  limit,
        "apikey": fmp_key,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{FMP_STABLE}/sec-filings-8k", params=params)
            resp.raise_for_status()
            filings = resp.json()
    except Exception as e:
        print(f"[sec-filings] context fetch failed: {e}")
        return ""

    if not filings:
        return f"No 8-K filings found in the last {days} days."

    lines = [f"8-K filings — last {days} days ({len(filings)} results):"]
    for f in filings:
        symbol      = f.get("symbol", "?")
        filed       = (f.get("filingDate") or "")[:10]
        accepted    = (f.get("acceptedDate") or "")[:10]
        link        = f.get("finalLink") or f.get("link") or ""
        has_fin     = " [+financials]" if f.get("hasFinancials") else ""
        lines.append(f"  {symbol} | Filed {filed} | Accepted {accepted}{has_fin} | {link}")

    return "\n".join(lines)


@app.get("/sec-filings/form-type")
async def sec_filings_by_form_type(
    form_type: str = Query(..., alias="formType", description="SEC form type e.g. 8-K, 10-K, 10-Q, 13D, 13F, S-1, DEF 14A"),
    from_date: str = Query(default=None, alias="from", description="Start date YYYY-MM-DD (default: 30 days ago)"),
    to_date:   str = Query(default=None, alias="to",   description="End date YYYY-MM-DD (default: today)"),
    page:      int = Query(default=0,    description="Page index"),
    limit:     int = Query(default=20,   description="Results per page (max 100)"),
):
    """
    Search SEC filings by form type (premium FMP endpoint).
    Supports 8-K, 10-K, 10-Q, 13D, 13F, S-1, DEF 14A, and other SEC form types.
    """
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY not configured")

    today = date.today()
    params = {
        "formType": form_type,
        "from":     from_date or (today - timedelta(days=30)).isoformat(),
        "to":       to_date   or today.isoformat(),
        "page":     page,
        "limit":    min(limit, 100),
        "apikey":   fmp_key,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{FMP_STABLE}/sec-filings-search/form-type", params=params)
            resp.raise_for_status()
            filings = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"FMP error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMP request failed: {e}")

    return {"filings": filings, "count": len(filings), "form_type": form_type, "params": params}


@app.get("/sec-filings/search")
async def sec_filings_company_search(
    company: str = Query(..., description="Company or entity name to search e.g. 'Berkshire', 'Apple'"),
):
    """
    Search for SEC filers by company or entity name (premium FMP endpoint).
    Returns CIK, SIC code, industry title, business address, and phone number.
    Use this to identify the correct entity before pulling their specific filings.
    """
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY not configured")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{FMP_STABLE}/sec-filings-company-search/name",
                params={"company": company, "apikey": fmp_key},
            )
            resp.raise_for_status()
            results = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"FMP error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMP request failed: {e}")

    return {"results": results, "count": len(results), "query": company}


async def _fmp_get(path: str, params: dict) -> list | dict:
    """Shared FMP GET helper for simple symbol-keyed endpoints."""
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY not configured")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{FMP_STABLE}/{path}", params={**params, "apikey": fmp_key})
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"FMP error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMP request failed: {e}")


# ── ETF endpoints ─────────────────────────────────────────────────────────────

@app.get("/etf/holdings")
async def etf_holdings(symbol: str = Query(..., description="ETF ticker e.g. SPY, VWO, QQQ")):
    """
    Full breakdown of securities held within an ETF or mutual fund (premium FMP endpoint).
    Returns each holding's asset ticker, name, ISIN, CUSIP, share count,
    weight percentage, and market value.
    """
    data = await _fmp_get("etf/holdings", {"symbol": symbol.upper()})
    return {"symbol": symbol.upper(), "holdings": data, "count": len(data) if isinstance(data, list) else 0}


@app.get("/etf/sector-weightings")
async def etf_sector_weightings(symbol: str = Query(..., description="ETF ticker e.g. SPY, VWO, QQQ")):
    """
    Sector allocation breakdown for an ETF (premium FMP endpoint).
    Returns each sector's weight percentage — essential for sector exposure
    analysis, portfolio risk management, and stress testing.
    """
    data = await _fmp_get("etf/sector-weightings", {"symbol": symbol.upper()})
    return {"symbol": symbol.upper(), "sectors": data}


@app.get("/etf/asset-exposure")
async def etf_asset_exposure(symbol: str = Query(..., description="Stock ticker to look up ETF holders e.g. AAPL, NVDA, MSFT")):
    """
    Reverse ETF lookup — discover which ETFs hold a specific stock (premium FMP endpoint).
    Pass a stock ticker (e.g. AAPL) and receive a list of all ETFs that hold it,
    with each ETF's share count, weight percentage, and market value of the position.
    Useful for understanding institutional ETF exposure to a single name and for
    estimating passive-flow impact when the stock moves.
    """
    data = await _fmp_get("etf/asset-exposure", {"symbol": symbol.upper()})
    return {"symbol": symbol.upper(), "etf_holders": data, "count": len(data) if isinstance(data, list) else 0}


@app.get("/etf/info")
async def etf_info(symbol: str = Query(..., description="ETF ticker e.g. SPY, VWO, QQQ")):
    """
    ETF and mutual fund metadata (premium FMP endpoint).
    Returns expense ratio, AUM, issuer, benchmark index, inception date,
    and other operational details useful for liquidity and due diligence.
    """
    data = await _fmp_get("etf/info", {"symbol": symbol.upper()})
    return {"symbol": symbol.upper(), "info": data}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream a chat completion from OpenRouter as SSE.
    Event format: { "content": "..." } per token, then { "done": true }.
    """
    messages = [{"role": "system", "content": req.system or FINANCE_SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        _agent_stream_and_save("general-chat", messages, req.model, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Chat UI with streaming, cancel button, and error display."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Equity Research AI</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
    header{padding:14px 24px;border-bottom:1px solid #1e2535;display:flex;align-items:center;gap:12px}
    header h1{font-size:1rem;font-weight:600;color:#93c5fd;flex:1}
    select{background:#1a1f2e;border:1px solid #1e2535;color:#e2e8f0;padding:6px 10px;border-radius:6px;font-size:.85rem}
    #messages{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:20px}
    .bubble{max-width:820px;line-height:1.7;font-size:.95rem;white-space:pre-wrap;word-break:break-word}
    .bubble.user{align-self:flex-end;background:#1e3a5f;padding:12px 16px;border-radius:14px 14px 2px 14px}
    .bubble.assistant{align-self:flex-start}
    .bubble.assistant .label{font-size:.72rem;color:#64748b;margin-bottom:4px}
    .bubble.error{color:#f87171;align-self:flex-start;font-size:.9rem}
    #samples{display:none;padding:0 24px 10px;gap:8px;flex-wrap:wrap}
    .chip{background:#1a1f2e;border:1px solid #2d3748;color:#93c5fd;padding:7px 13px;border-radius:20px;font-size:.8rem;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:320px}
    .chip:hover{background:#1e3a5f;border-color:#3b82f6}
    #bar{display:flex;gap:8px;padding:14px 24px;border-top:1px solid #1e2535;align-items:center}
    #input{flex:1;background:#1a1f2e;border:1px solid #1e2535;color:#e2e8f0;padding:12px 16px;border-radius:8px;font-size:.95rem;outline:none;resize:none;height:48px}
    #input:focus{border-color:#3b82f6}
    .btn{border:none;padding:0 18px;height:48px;border-radius:8px;font-weight:600;cursor:pointer;font-size:.9rem}
    #send{background:#3b82f6;color:#fff}
    #cancel{background:#374151;color:#e2e8f0;display:none}
    #send:disabled,#cancel:disabled{opacity:.4;cursor:not-allowed}
    .thinking{color:#64748b;font-style:italic}
    .sym-btn{background:#1e3a5f;border:1px solid #3b82f6;color:#93c5fd;padding:6px 14px;border-radius:6px;font-size:.8rem;font-weight:600;cursor:pointer;white-space:nowrap}
    .sym-btn:hover{background:#243f6a;border-color:#60a5fa}
    #symModal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;align-items:center;justify-content:center}
    #symModal.open{display:flex}
    #symBox{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;width:min(780px,95vw);max-height:82vh;display:flex;flex-direction:column;overflow:hidden}
    #symHeader{display:flex;align-items:center;gap:10px;padding:16px 20px;border-bottom:1px solid #2d3748}
    #symTitle{color:#93c5fd;font-weight:600;font-size:.95rem;white-space:nowrap}
    #symSearch{flex:1;background:#0f1117;border:1px solid #2d3748;color:#e2e8f0;padding:8px 12px;border-radius:6px;font-size:.9rem;outline:none}
    #symSearch:focus{border-color:#3b82f6}
    #symClose{background:none;border:none;color:#64748b;font-size:1.5rem;cursor:pointer;line-height:1;padding:0 4px}
    #symClose:hover{color:#e2e8f0}
    #symCount{font-size:.75rem;color:#64748b;padding:6px 20px;border-bottom:1px solid #1e2535}
    #symList{overflow-y:auto;flex:1}
    #symList table{width:100%;border-collapse:collapse;font-size:.85rem}
    #symList th{position:sticky;top:0;background:#1a1f2e;color:#64748b;font-weight:500;text-align:left;padding:8px 16px;border-bottom:1px solid #2d3748;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
    #symList td{padding:9px 16px;border-bottom:1px solid #1e2535}
    #symList tbody tr{cursor:pointer}
    #symList tbody tr:hover td{background:#0f1117}
    #symList td:first-child{color:#93c5fd;font-weight:600;font-family:monospace;font-size:.9rem}
    #symLoading{padding:32px;text-align:center;color:#64748b;font-style:italic}
  </style>
</head>
<body>
  <header>
    <h1>Equity Research AI</h1>
    <select id="agent" onchange="onAgentChange()">
      <option value="">— General chat —</option>
    </select>
    <select id="model">
      <option value="nemotron">Nemotron Ultra 550B</option>
      <option value="laguna">Laguna M.1</option>
    </select>
    <button class="sym-btn" onclick="openSymbols()">Browse Symbols</button>
  </header>
  <div id="messages"></div>
  <div id="samples"></div>
  <div id="bar">
    <textarea id="input" placeholder="VIX is at 28 and rising — what does elevated fear mean for equity positioning and sector rotation today?"></textarea>
    <button id="cancel" class="btn" onclick="cancel()">Cancel</button>
    <button id="send"   class="btn" onclick="sendMsg()">Send</button>
  </div>

  <!-- Symbols modal -->
  <div id="symModal" onclick="if(event.target===this)closeSymbols()">
    <div id="symBox">
      <div id="symHeader">
        <span id="symTitle">Available Symbols — FMP Free Tier</span>
        <input id="symSearch" type="text" placeholder="Search ticker or company name…" oninput="debounceFilter()">
        <button id="symClose" onclick="closeSymbols()" title="Close">&#x2715;</button>
      </div>
      <div id="symCount"></div>
      <div id="symList">
        <div id="symLoading">Loading symbols…</div>
        <table style="display:none"><thead><tr><th>Ticker</th><th>Company</th><th>Exchange</th><th>Type</th></tr></thead><tbody id="symBody"></tbody></table>
      </div>
    </div>
  </div>

  <script>
    const inputEl  = document.getElementById('input');
    const sendBtn  = document.getElementById('send');
    const cancelBtn= document.getElementById('cancel');
    const msgs     = document.getElementById('messages');
    let controller = null;
    let timerInterval = null;

    const AGENT_META = {
      '': {
        placeholder: 'VIX is at 28 and rising — what does elevated fear mean for equity positioning and sector rotation today?',
        samples: []
      },
      'chief-capital-modelling-agent': {
        placeholder: 'SCR calibration for AI model failure liability with no loss history (EVT tail-fitting)',
        samples: [
          'We are writing AI model failure liability for enterprise SaaS startups with no credible loss history. How should we calibrate the SCR frequency-severity distribution under Solvency II Pillar 1, and what EVT tail-fitting approach would you recommend?',
          'Our AI errors & omissions book covers 200 Series B/C startups at £150M GWP. Design a CAT XL reinsurance programme that maximises SCR relief within a 15% net cost-of-capital constraint — include attachment point, limit, and reinstatement recommendations.',
          'Model a systemic AI infrastructure failure scenario: a major cloud provider LLM API outage affects 80% of our insured AI startup portfolio simultaneously. Quantify the aggregate PML, SCR uplift, Solvency II coverage ratio impact, and credible management actions.',
          'We want to launch an AI product liability line targeting Series A startups at £1M–£5M limit. Using Euler allocation, what capital per £1M of limit deployed justifies a RORAC above our 12% hurdle rate, and how does this compare against our existing cyber book?'
        ]
      },
      'transactional-liability-wi-agent': {
        placeholder: 'e.g. We are underwriting a W&I policy on a Series C AI startup acquisition — what are the key warranty risks to stress-test?',
        samples: [
          'We are underwriting a W&I policy for a £45M acquisition of an AI data analytics startup. The SPA includes IP ownership warranties and data privacy representations. What are the three highest-severity warranty risks and how should we structure the tipping basket?',
          'Price a W&I policy for a fintech AI startup acquisition at £80M enterprise value. The target processes personal financial data under GDPR. Provide a ROL indication, recommended retention, and key exclusions given the regulatory exposure.',
          'The acquirer of an AI startup is claiming a material warranty breach six months post-close — the training data allegedly included unlicensed third-party content, creating IP infringement exposure. Walk through the claims investigation process and reserve methodology.',
          'We have a pipeline of five AI startup W&I deals closing this quarter totalling £220M in aggregate limit. Assess the portfolio accumulation risk, identify correlated exposures across the book, and recommend a reinsurance structure to manage peak aggregate loss.'
        ]
      },
      'sec-filings-analyst': {
        placeholder: 'Which companies filed material 8-K events this week — M&A, CEO changes, or impairments — and what are the investment implications?',
        samples: [
          'Triage the recent 8-K filings above. Which events are Tier 1 catalysts (M&A close, CEO departure, bankruptcy) and which can be deprioritised? Give me a ranked watchlist with investment action for each.',
          'Flag all Item 5.02 leadership change filings. Which CEO or CFO departures look sudden and unexplained vs planned succession, and what does each signal about corporate health or a pending strategic pivot?',
          'I want to screen for activist investor activity. Walk me through how to use 13D filings to identify hedge funds accumulating stakes, and what to look for in the Schedule 13D to assess whether an activist campaign is likely.',
          'Explain the key differences between 10-K, 10-Q, 8-K, 13D, and 13F filings — when each is filed, what it discloses, and which form type is most relevant for spotting early-stage investment catalysts.'
        ]
      }
    };

    // Populate agent dropdown
    fetch('/agents').then(r => r.json()).then(({ agents }) => {
      const sel = document.getElementById('agent');
      agents.forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.name;
        opt.textContent = a.name.replace(/-agent$/, '').replace(/-/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
        sel.appendChild(opt);
      });
    });

    function onAgentChange() {
      const name = document.getElementById('agent').value;
      const meta = AGENT_META[name] || AGENT_META[''];
      inputEl.placeholder = meta.placeholder;
      renderSamples(meta.samples);
    }

    function renderSamples(samples) {
      const box = document.getElementById('samples');
      box.innerHTML = '';
      if (!samples.length) { box.style.display = 'none'; return; }
      box.style.display = 'flex';
      samples.forEach(s => {
        const chip = document.createElement('button');
        chip.className = 'chip';
        chip.title = s;
        chip.textContent = s.length > 72 ? s.slice(0, 70) + '…' : s;
        chip.onclick = () => { inputEl.value = s; inputEl.focus(); };
        box.appendChild(chip);
      });
    }

    function getEndpoint() {
      const agent = document.getElementById('agent').value;
      return agent ? `/agents/${agent}/chat/stream` : '/chat/stream';
    }

    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
    });

    function addBubble(role, text='') {
      const wrap = document.createElement('div');
      wrap.className = 'bubble ' + role;
      if (role === 'assistant') {
        const lbl = document.createElement('div');
        lbl.className = 'label';
        lbl.textContent = document.getElementById('model').selectedOptions[0].text;
        wrap.appendChild(lbl);
      }
      const body = document.createElement('div');
      body.textContent = text;
      wrap.appendChild(body);
      msgs.appendChild(wrap);
      msgs.scrollTop = msgs.scrollHeight;
      return body;
    }

    function setBusy(busy) {
      sendBtn.disabled = busy;
      cancelBtn.style.display = busy ? 'inline-block' : 'none';
      inputEl.disabled = busy;
    }

    function startTimer(el) {
      const t0 = Date.now();
      timerInterval = setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000);
        const m = Math.floor(s / 60);
        const ss = String(s % 60).padStart(2, '0');
        el.textContent = m > 0 ? `Waiting… ${m}m${ss}s` : `Waiting… ${s}s`;
      }, 1000);
    }

    function stopTimer() {
      clearInterval(timerInterval);
      timerInterval = null;
    }

    function cancel() {
      if (controller) { controller.abort(); controller = null; }
      stopTimer();
    }

    // ── Symbols modal ────────────────────────────────────────────────────────
    let allSymbols = [];
    let symLoaded  = false;
    let filterTimer = null;

    function openSymbols() {
      document.getElementById('symModal').classList.add('open');
      document.getElementById('symSearch').focus();
      if (!symLoaded) loadSymbols();
    }

    function closeSymbols() {
      document.getElementById('symModal').classList.remove('open');
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeSymbols();
    });

    async function loadSymbols() {
      try {
        const res = await fetch('/symbols');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { symbols } = await res.json();
        allSymbols = symbols || [];
        symLoaded  = true;
        document.getElementById('symLoading').style.display = 'none';
        document.querySelector('#symList table').style.display = '';
        document.getElementById('symBody').addEventListener('click', e => {
          const row = e.target.closest('tr[data-ticker]');
          if (row) pickSymbol(row.dataset.ticker, row.dataset.name);
        });
        renderSymbols(allSymbols);
      } catch(e) {
        document.getElementById('symLoading').textContent = 'Failed to load: ' + e.message;
      }
    }

    function debounceFilter() {
      clearTimeout(filterTimer);
      filterTimer = setTimeout(applyFilter, 180);
    }

    function applyFilter() {
      const q = document.getElementById('symSearch').value.toLowerCase().trim();
      const filtered = q
        ? allSymbols.filter(s =>
            (s.symbol||'').toLowerCase().includes(q) ||
            (s.name||'').toLowerCase().includes(q))
        : allSymbols;
      renderSymbols(filtered);
    }

    function esc(s) {
      return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function renderSymbols(list) {
      const shown = list.slice(0, 250);
      document.getElementById('symCount').textContent =
        `Showing ${shown.length} of ${list.length.toLocaleString()} (${allSymbols.length.toLocaleString()} total)`;
      document.getElementById('symBody').innerHTML = shown.map(s =>
        `<tr data-ticker="${esc(s.symbol)}" data-name="${esc(s.name)}">
          <td>${esc(s.symbol)}</td>
          <td>${esc(s.name)}</td>
          <td>${esc(s.exchangeShortName||s.exchange)}</td>
          <td>${esc(s.type)}</td>
        </tr>`
      ).join('');
    }

    function pickSymbol(ticker, name) {
      const label = name ? `${ticker} (${name})` : ticker;
      inputEl.value = msgs.children.length === 0 ? `Research ${label}` : label;
      closeSymbols();
      inputEl.focus();
    }
    // ── End symbols modal ─────────────────────────────────────────────────────

    async function sendMsg() {
      const message = inputEl.value.trim();
      if (!message || sendBtn.disabled) return;
      inputEl.value = '';
      setBusy(true);
      addBubble('user', message);
      const out = addBubble('assistant', '');
      out.className = 'thinking';
      out.textContent = 'Waiting… 0s';
      startTimer(out);

      controller = new AbortController();
      try {
        const res = await fetch(getEndpoint(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, model: document.getElementById('model').value }),
          signal: controller.signal,
        });
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let started = false;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\\n');
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.error) {
              stopTimer();
              out.className = '';
              out.closest('.bubble').className = 'bubble error';
              out.textContent = '⚠ ' + data.error;
              return;
            }
            if (data.content) {
              if (!started) { stopTimer(); out.className=''; out.textContent=''; started=true; }
              out.textContent += data.content;
              msgs.scrollTop = msgs.scrollHeight;
            }
          }
        }
        if (!started) { stopTimer(); out.className=''; out.textContent = '(no response)'; }
      } catch (e) {
        stopTimer();
        out.className = '';
        out.closest('.bubble').className = 'bubble error';
        out.textContent = e.name === 'AbortError' ? 'Cancelled.' : '⚠ ' + e.message;
      } finally {
        controller = null;
        setBusy(false);
        inputEl.focus();
      }
    }
  </script>
</body>
</html>"""
