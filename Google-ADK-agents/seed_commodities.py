#!/usr/bin/env python3
"""Seed the A2A store with commodities notes across coal / gas / oil / energy.

There is ONE commodities agent (OilPrice tools, OpenRouter — no Gemini); we run
it once per **theme** so each run becomes a distinct document for semantic
recall: its own `agent_runs` envelope, `prices` rows (from the tool fetches), and
a narrative `agent_responses` note.

    uv run python seed_commodities.py            # all four themes
    uv run python seed_commodities.py oil gas    # a subset

Reads keys from finance_coordinator/.env. Persistence is best-effort: if the
SUPABASE_* vars are unset the agent still runs, it just writes no rows.
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("LITELLM_LOG", "ERROR")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

PROJECT = Path(__file__).resolve().parent  # Google-ADK-agents/

# One themed prompt per commodity family. Each becomes a separate run/document.
THEMES: dict[str, str] = {
    "coal": (
        "What is Newcastle coal (API6) trading at right now, and how has it "
        "moved over the past month? Give a short read."
    ),
    "gas": (
        "What are UK natural gas and TTF natural gas spot prices right now, and "
        "what is the recent trend? Give a short read."
    ),
    "oil": (
        "What are Brent crude and WTI crude prices right now, and how have they "
        "moved this week? Give a short read."
    ),
    "energy": (
        "Show the Fuse Energy watchlist prices and give a short read on the "
        "overall energy complex (gas, petroleum, carbon, coal)."
    ),
}


async def run_one(theme: str, prompt: str) -> None:
    """Run the commodities agent for one theme and persist run + prices + note."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from a2a_finance import storage
    from commodities_agent.agent import root_agent

    run_id = storage.start_run("commodities_agent", subject=theme.upper(), prompt=prompt)
    print(f"\n=== [{theme}] run_id={run_id} ===", file=sys.stderr)

    runner = InMemoryRunner(agent=root_agent, app_name="seed")
    await runner.session_service.create_session(
        app_name="seed", user_id="u", session_id=theme
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    final = None
    async for ev in runner.run_async(user_id="u", session_id=theme, new_message=msg):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    print(f"  [tool] {p.function_call.name}", file=sys.stderr)
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))

    print((final or "(no response)")[:400])
    if final:
        storage.save_response(
            "commodities_agent", subject=theme.upper(), text=final, run_id=run_id
        )


async def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / "finance_coordinator" / ".env")
    sys.path.insert(0, str(PROJECT))

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1
    if not os.getenv("OILPRICE_API_KEY"):
        print("Warning: OILPRICE_API_KEY not set — price tools will return errors.", file=sys.stderr)

    wanted = [a.lower() for a in sys.argv[1:]] or list(THEMES)
    unknown = [w for w in wanted if w not in THEMES]
    if unknown:
        print(f"Unknown theme(s) {unknown}; valid: {list(THEMES)}", file=sys.stderr)
        return 1

    from a2a_finance import storage

    if not storage.enabled():
        print("▸ persistence disabled (no SUPABASE_* env) — running without writing rows.",
              file=sys.stderr)

    for theme in wanted:
        await run_one(theme, THEMES[theme])
    print(f"\n▸ seeded {len(wanted)} commodity note(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
