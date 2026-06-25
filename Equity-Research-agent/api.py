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
from datetime import date
from pathlib import Path

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR  = Path(__file__).parent
AGENTS_DIR = BASE_DIR / "agents"
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

    # ── Supabase insert ───────────────────────────────────────────────────────
    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO agent_responses (agent_name, model, query, response)
                       VALUES ($1, $2, $3, $4)""",
                    agent_name, full_model, query, response,
                )
        except Exception as e:
            print(f"[db] save failed: {e}")

    # ── markdown file ─────────────────────────────────────────────────────────
    out_dir = BASE_DIR / "sample-outputs"
    out_dir.mkdir(exist_ok=True)
    base = f"{agent_name}-{today}"
    path = out_dir / f"{base}.md"
    counter = 1
    while path.exists():
        path = out_dir / f"{base}-{counter}.md"
        counter += 1
    header = (
        f"# Agent Response — {agent_name}\n\n"
        f"**Query:** {query}\n\n"
        f"**Agent:** {agent_name}\n"
        f"**Model:** {full_model}\n"
        f"**Date:** {today}\n\n"
        f"---\n\n"
    )
    path.write_text(header + response + "\n", encoding="utf-8")
    print(f"[save] {path.name}")


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


# ── routes ────────────────────────────────────────────────────────────────────

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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": req.message},
    ]
    return StreamingResponse(
        _agent_stream_and_save(name, messages, req.model, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _agent_stream_and_save(agent_name: str, messages: list, model: str, query: str):
    """Wraps _openrouter_stream: passes tokens through and saves the full response on completion."""
    accumulated: list[str] = []
    async for chunk in _openrouter_stream(messages, model):
        yield chunk
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                if "content" in data:
                    accumulated.append(data["content"])
                elif data.get("done") and accumulated:
                    asyncio.create_task(
                        _save_agent_response(agent_name, model, query, "".join(accumulated))
                    )
            except (json.JSONDecodeError, KeyError):
                pass


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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream a chat completion from OpenRouter as SSE.
    Event format: { "content": "..." } per token, then { "done": true }.
    """
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        _openrouter_stream(messages, req.model),
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
  </header>
  <div id="messages"></div>
  <div id="samples"></div>
  <div id="bar">
    <textarea id="input" placeholder="VIX is at 28 and rising — what does elevated fear mean for equity positioning and sector rotation today?"></textarea>
    <button id="cancel" class="btn" onclick="cancel()">Cancel</button>
    <button id="send"   class="btn" onclick="sendMsg()">Send</button>
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
