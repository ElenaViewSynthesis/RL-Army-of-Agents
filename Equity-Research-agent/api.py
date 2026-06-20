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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
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
