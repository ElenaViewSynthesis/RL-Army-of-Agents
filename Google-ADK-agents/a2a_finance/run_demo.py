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
PORT = os.getenv("A2A_VALUATION_PORT", "8001")
CARD = f"http://localhost:{PORT}/.well-known/agent-card.json"

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT / "finance_coordinator" / ".env")
sys.path.insert(0, str(PROJECT))


def start_server() -> subprocess.Popen:
    return subprocess.Popen(
        ["uv", "run", "uvicorn", "a2a_finance.valuation_service:a2a_app",
         "--port", PORT, "--log-level", "warning"],
        cwd=str(PROJECT), env=os.environ.copy(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_for_card(timeout: float = 60.0) -> bool:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(CARD, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


async def run_coordinator(ticker: str, question: str) -> None:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from a2a_finance.coordinator import root_agent

    prompt = f"For {ticker}: {question}" if question else f"Is {ticker} cheap or expensive right now?"
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
    print("\n--- COORDINATOR (via A2A) ---\n")
    print(final or "(no response)")


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    question = " ".join(sys.argv[2:]).strip()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        sys.exit(1)

    print(f"▸ starting A2A valuation service on :{PORT} …", file=sys.stderr)
    server = start_server()
    try:
        if not wait_for_card():
            print("A2A service did not come up in time.", file=sys.stderr)
            sys.exit(2)
        print("▸ service up; running coordinator\n", file=sys.stderr)
        asyncio.run(run_coordinator(ticker, question))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
