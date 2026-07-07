#!/usr/bin/env python3
"""Run the reasoning-enabled valuation agent standalone.

The valuation_agent runs on the nemotron reasoning model via OpenRouter, so it
works without Gemini quota. This drives it directly (bypassing the Gemini
coordinator) and prints tool calls + the final verdict.

Examples (WSL / Linux / macOS / Windows):
    uv run python run_valuation.py NVDA
    uv run python run_valuation.py AAPL -q "is it cheap vs its DCF?"
    uv run python run_valuation.py TSLA --model deepseek/deepseek-chat --no-reasoning
    uv run python run_valuation.py MSFT --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252; model output may contain other Unicode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Resolve paths relative to THIS file so the script runs from any working
# directory (important under WSL where you may cd elsewhere).
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_valuation.py",
        description="Run the OpenRouter reasoning valuation agent on a ticker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ticker", help="Stock ticker, e.g. NVDA")
    p.add_argument(
        "-q", "--question", default="",
        help="Specific question to ask (default: general cheap/expensive verdict)",
    )
    p.add_argument(
        "-m", "--model", default=None,
        help="Override OPENROUTER_MODEL (e.g. deepseek/deepseek-chat)",
    )
    p.add_argument(
        "--reasoning", dest="reasoning", action=argparse.BooleanOptionalAction,
        default=None, help="Force reasoning on/off (--reasoning / --no-reasoning)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON (ticker, model, reasoning, tools, verdict)",
    )
    return p.parse_args()


async def run(args: argparse.Namespace) -> int:
    # Apply CLI overrides to the environment BEFORE importing the agent, since
    # config.py reads these at import time to build the model handle.
    if args.model is not None:
        os.environ["OPENROUTER_MODEL"] = args.model
    if args.reasoning is not None:
        os.environ["OPENROUTER_REASONING"] = "1" if args.reasoning else "0"

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from finance_coordinator.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING
    from finance_coordinator.sub_agents.valuation_agent import valuation_agent

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1

    ticker = args.ticker.upper()
    prompt = (
        f"For {ticker}: {args.question}"
        if args.question
        else f"Is {ticker} cheap or expensive? Use your tools, then give a one-paragraph verdict."
    )

    if not args.json:
        print(f"model: {OPENROUTER_MODEL_ID} | reasoning: {OPENROUTER_REASONING}\n")

    runner = InMemoryRunner(agent=valuation_agent, app_name="valuation")
    await runner.session_service.create_session(
        app_name="valuation", user_id="u", session_id="s"
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tools_called: list[dict] = []
    final: str | None = None
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "function_call", None):
                    call = {"name": part.function_call.name, "args": dict(part.function_call.args)}
                    tools_called.append(call)
                    if not args.json:
                        print(f"  [tool] {call['name']}({call['args']})")
        if ev.is_final_response() and ev.content:
            final = "".join(pt.text for pt in ev.content.parts if getattr(pt, "text", None))

    if args.json:
        print(json.dumps({
            "ticker": ticker,
            "model": OPENROUTER_MODEL_ID,
            "reasoning": OPENROUTER_REASONING,
            "tools": tools_called,
            "verdict": final,
        }, indent=2))
    else:
        print("\n--- VERDICT ---\n")
        print(final or "(no text response)")
    return 0


def main() -> None:
    args = parse_args()
    # Load env from the package's .env regardless of CWD.
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / "finance_coordinator" / ".env")
    # Make sub_agent imports resolve when run from another directory.
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
