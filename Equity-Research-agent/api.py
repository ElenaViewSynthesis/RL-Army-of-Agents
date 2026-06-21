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
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
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


async def _openrouter_stream(messages: list, model: str):
    full_model = MODEL_ALIASES.get(model, model)
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={"model": full_model, "messages": messages, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                except (KeyError, json.JSONDecodeError):
                    continue
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
    """Simple chat UI with streaming support."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Equity Research AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
    header { padding: 14px 24px; border-bottom: 1px solid #1e2535; display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 1rem; font-weight: 600; color: #93c5fd; }
    select { background: #1a1f2e; border: 1px solid #1e2535; color: #e2e8f0; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; }
    #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
    .bubble { max-width: 820px; line-height: 1.7; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
    .bubble.user { align-self: flex-end; background: #1e3a5f; padding: 12px 16px; border-radius: 14px 14px 2px 14px; }
    .bubble.assistant { align-self: flex-start; color: #e2e8f0; }
    .bubble.assistant .label { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }
    #bar { display: flex; gap: 8px; padding: 14px 24px; border-top: 1px solid #1e2535; }
    #input { flex: 1; background: #1a1f2e; border: 1px solid #1e2535; color: #e2e8f0; padding: 12px 16px; border-radius: 8px; font-size: 0.95rem; outline: none; resize: none; height: 48px; }
    #input:focus { border-color: #3b82f6; }
    #send { background: #3b82f6; color: #fff; border: none; padding: 0 20px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; }
    #send:disabled { opacity: 0.4; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>
    <h1>Equity Research AI</h1>
    <select id="model">
      <option value="nemotron">Nemotron Ultra 550B</option>
      <option value="laguna">Laguna M.1</option>
    </select>
  </header>
  <div id="messages"></div>
  <div id="bar">
    <textarea id="input" placeholder="Ask about a stock, sector, or market event…"></textarea>
    <button id="send" onclick="send()">Send</button>
  </div>

  <script>
    const input = document.getElementById('input');
    const btn   = document.getElementById('send');
    const msgs  = document.getElementById('messages');

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    function append(role, text) {
      const wrap = document.createElement('div');
      wrap.className = `bubble ${role}`;
      if (role === 'assistant') {
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = document.getElementById('model').selectedOptions[0].text;
        wrap.appendChild(label);
      }
      const body = document.createElement('div');
      body.textContent = text;
      wrap.appendChild(body);
      msgs.appendChild(wrap);
      msgs.scrollTop = msgs.scrollHeight;
      return body;
    }

    async function send() {
      const message = input.value.trim();
      if (!message || btn.disabled) return;
      input.value = '';
      btn.disabled = true;
      append('user', message);
      const out = append('assistant', '');

      try {
        const res = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, model: document.getElementById('model').value }),
        });
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\\n');
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.content) { out.textContent += data.content; msgs.scrollTop = msgs.scrollHeight; }
          }
        }
      } catch (e) {
        out.textContent = 'Error: ' + e.message;
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }
  </script>
</body>
</html>"""
