#!/usr/bin/env python3
"""Run the reasoning valuation agent — streaming by default.

Primary mode streams the reasoning model's answer token by token. `--structured`
runs a two-step flow (tools-gathering agent -> schema formatter) and prints a
Pydantic-validated ResearchNote as JSON. The valuation agent runs on OpenRouter,
so it works without Gemini quota.

Examples (WSL / Linux / macOS / Windows):
    uv run python run_valuation.py NVDA                      # streaming (default)
    uv run python run_valuation.py AAPL -q "cheap vs DCF?"   # streaming, specific
    uv run python run_valuation.py TSLA --structured          # zod/Pydantic note (JSON)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("LITELLM_LOG", "ERROR")  # quiet litellm's info/provider noise

# Windows consoles default to cp1252; model output may contain other Unicode.
# No-op on WSL/Linux where stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_valuation.py",
        description="Stream the OpenRouter reasoning valuation agent (or emit a structured note).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ticker", help="Stock ticker, e.g. NVDA")
    p.add_argument("-q", "--question", default="", help="Specific question to ask")
    p.add_argument("-m", "--model", default=None, help="Override OPENROUTER_MODEL")
    p.add_argument(
        "--reasoning", dest="reasoning", action=argparse.BooleanOptionalAction,
        default=None, help="Force reasoning on/off",
    )
    p.add_argument(
        "--structured", "--json", dest="structured", action="store_true",
        help="Emit a schema-validated ResearchNote as JSON (two-step)",
    )
    return p.parse_args()


def _prompt(ticker: str, question: str) -> str:
    return (
        f"For {ticker}: {question}"
        if question
        else f"Is {ticker} cheap or expensive? Use your tools, then give a one-paragraph verdict."
    )


async def stream_run(args: argparse.Namespace) -> int:
    """Primary mode: stream the valuation agent's answer token by token."""
    from google.adk.runners import InMemoryRunner
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.genai import types

    from finance_coordinator.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING
    from finance_coordinator.sub_agents.valuation_agent import valuation_agent

    ticker = args.ticker.upper()
    print(f"model: {OPENROUTER_MODEL_ID} | reasoning: {OPENROUTER_REASONING}\n", file=sys.stderr)

    runner = InMemoryRunner(agent=valuation_agent, app_name="valuation")
    await runner.session_service.create_session(app_name="valuation", user_id="u", session_id="s")
    msg = types.Content(role="user", parts=[types.Part(text=_prompt(ticker, args.question))])

    async for ev in runner.run_async(
        user_id="u", session_id="s", new_message=msg,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "function_call", None):
                    print(f"\n  [tool] {part.function_call.name}({dict(part.function_call.args)})", file=sys.stderr)
                elif getattr(part, "text", None) and ev.partial:
                    sys.stdout.write(part.text)
                    sys.stdout.flush()
    print()
    return 0


async def structured_run(args: argparse.Namespace) -> int:
    """Two-step: valuation agent gathers/analyzes, formatter emits ResearchNote."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from finance_coordinator.sub_agents.valuation_agent import valuation_agent
    from finance_coordinator.sub_agents.note_formatter import note_formatter
    from finance_coordinator.schema import ResearchNote

    ticker = args.ticker.upper()

    # Step 1 — gather + analyze with tools (non-streamed; we need the full text).
    r1 = InMemoryRunner(agent=valuation_agent, app_name="v1")
    await r1.session_service.create_session(app_name="v1", user_id="u", session_id="s")
    msg = types.Content(role="user", parts=[types.Part(text=_prompt(ticker, args.question))])
    analysis = ""
    async for ev in r1.run_async(user_id="u", session_id="s", new_message=msg):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "function_call", None):
                    print(f"  [tool] {part.function_call.name}", file=sys.stderr)
        if ev.is_final_response() and ev.content:
            analysis = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))

    # Step 2 — format the analysis into the structured note.
    r2 = InMemoryRunner(agent=note_formatter, app_name="v2")
    await r2.session_service.create_session(app_name="v2", user_id="u", session_id="s")
    fmt_input = types.Content(
        role="user",
        parts=[types.Part(text=f"Analyst notes for {ticker}:\n\n{analysis}")],
    )
    note_json = None
    async for ev in r2.run_async(user_id="u", session_id="s", new_message=fmt_input):
        if ev.is_final_response() and ev.content:
            note_json = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))

    if not note_json:
        print("Formatter produced no output.", file=sys.stderr)
        return 2
    note = ResearchNote.model_validate_json(note_json)
    print(note.model_dump_json(indent=2))
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.model is not None:
        os.environ["OPENROUTER_MODEL"] = args.model
    if args.reasoning is not None:
        os.environ["OPENROUTER_REASONING"] = "1" if args.reasoning else "0"
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1
    return await (structured_run(args) if args.structured else stream_run(args))


def main() -> None:
    args = parse_args()
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / "finance_coordinator" / ".env")
    sys.path.insert(0, str(SCRIPT_DIR))
    from a2a_finance import observability
    observability.init_tracing()
    try:
        code = asyncio.run(run(args))
    finally:
        observability.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
