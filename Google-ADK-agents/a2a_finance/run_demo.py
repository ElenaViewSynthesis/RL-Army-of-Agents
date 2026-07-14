#!/usr/bin/env python3
"""Self-contained A2A demo.

Boots the valuation A2A service as a subprocess, waits for its agent card, then
runs the coordinator — which delegates to the remote agent over A2A — and tears
the service down. Everything runs on the OpenRouter client SDK (no Gemini).

    uv run python a2a_finance/run_demo.py NVDA
    uv run python a2a_finance/run_demo.py AAPL "is it cheap vs its DCF?"
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent          # a2a_finance/
PROJECT = SCRIPT_DIR.parent                            # Google-ADK-agents/

# Load .env BEFORE reading any A2A_*_PORT overrides below, so the port uvicorn
# binds matches the port each service module advertises in its agent card.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT / "finance_coordinator" / ".env")
sys.path.insert(0, str(PROJECT))

# (module app, port) for each specialist A2A service.
SERVICES = [
    ("a2a_finance.fundamentals_service:a2a_app", os.getenv("A2A_FUNDAMENTALS_PORT", "8002")),
    ("a2a_finance.valuation_service:a2a_app", os.getenv("A2A_VALUATION_PORT", "8001")),
    ("a2a_finance.risk_service:a2a_app", os.getenv("A2A_RISK_PORT", "8003")),
    ("a2a_finance.commodities_service:a2a_app", os.getenv("A2A_COMMODITIES_PORT", "8004")),
]


def start_servers(run_id: str | None = None) -> list[subprocess.Popen]:
    procs = []
    for app, port in SERVICES:
        env = os.environ.copy()
        if run_id:
            env["A2A_RUN_ID"] = run_id
        procs.append(subprocess.Popen(
            ["uv", "run", "uvicorn", app, "--port", port, "--log-level", "warning"],
            cwd=str(PROJECT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    return procs


def wait_for_all(timeout: float = 90.0) -> bool:
    import httpx
    cards = [f"http://localhost:{port}/.well-known/agent-card.json" for _, port in SERVICES]
    deadline = time.time() + timeout
    pending = set(cards)
    while pending and time.time() < deadline:
        for url in list(pending):
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    pending.discard(url)
            except Exception:
                pass
        if pending:
            time.sleep(1)
    return not pending


async def run_coordinator(
    ticker: str,
    question: str,
    *,
    research: bool = False,
    run_id: str | None = None,
) -> None:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    if research:
        # Fan-out: consult all specialists (as AgentTools) → synthesize one note.
        from a2a_finance.research import root_agent
        prompt = f"Write a full research note on {ticker}."
        label = "RESEARCH NOTE (fan-out via A2A)"
    else:
        # Routing: coordinator transfers the query to one specialist.
        from a2a_finance.coordinator import root_agent
        prompt = f"For {ticker}: {question}" if question else f"Is {ticker} cheap or expensive right now?"
        label = "COORDINATOR (via A2A)"
    runner = InMemoryRunner(agent=root_agent, app_name="a2a")
    await runner.session_service.create_session(app_name="a2a", user_id="u", session_id="s")
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    final = None
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    print(f"  [{ev.author}] -> {p.function_call.name}", file=sys.stderr)
        if ev.is_final_response() and ev.content:
            t = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
            if t:
                final = t
    print(f"\n--- {label} ---\n")
    print(final or "(no response)")
    if final:
        from a2a_finance import storage

        storage.save_response(
            "research_agent" if research else "finance_coordinator",
            subject=ticker,
            text=final,
            run_id=run_id,
        )


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--research", "--fanout")]
    research = any(a in ("--research", "--fanout") for a in sys.argv[1:])
    ticker = (args[0] if args else "NVDA").upper()
    question = " ".join(args[1:]).strip()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        sys.exit(1)

    if research:
        prompt = f"Write a full research note on {ticker}."
    else:
        prompt = f"For {ticker}: {question}" if question else f"Is {ticker} cheap or expensive right now?"
    from a2a_finance import storage

    run_id = storage.start_run(
        "research_agent" if research else "finance_coordinator",
        subject=ticker,
        prompt=prompt,
    )
    if run_id:
        os.environ["A2A_RUN_ID"] = run_id
        print(f"▸ persistence run_id={run_id}", file=sys.stderr)
    elif storage.enabled():
        print("▸ persistence configured, but start_run did not return an id", file=sys.stderr)
    else:
        print("▸ persistence disabled; no DB rows will be written", file=sys.stderr)

    ports = ", ".join(port for _, port in SERVICES)
    print(f"▸ starting A2A services on :{ports} …", file=sys.stderr)
    servers = start_servers(run_id)
    try:
        if not wait_for_all():
            print("Not all A2A services came up in time.", file=sys.stderr)
            sys.exit(2)
        mode = "research fan-out" if research else "coordinator"
        print(f"▸ all services up; running {mode}\n", file=sys.stderr)
        asyncio.run(run_coordinator(ticker, question, research=research, run_id=run_id))
    finally:
        for s in servers:
            s.terminate()
        for s in servers:
            try:
                s.wait(timeout=10)
            except Exception:
                s.kill()


if __name__ == "__main__":
    main()
