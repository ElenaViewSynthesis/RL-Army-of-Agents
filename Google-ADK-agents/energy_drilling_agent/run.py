#!/usr/bin/env python3
"""Run the energy drilling agent standalone (streaming answer).

    uv run python -m energy_drilling_agent.run "How many DUC wells in the Permian?"
    uv run python -m energy_drilling_agent.run "Which basins have rising DUC counts?"

Runs on OpenRouter (no Gemini needed); reads keys from finance_coordinator/.env.
The DPR endpoints are PREMIUM (OilPrice 'Scale' plan) — on the free tier the agent
will report that the dataset requires an upgrade rather than invent figures.
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

PROJECT = Path(__file__).resolve().parent.parent  # Google-ADK-agents/


async def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / "finance_coordinator" / ".env")
    sys.path.insert(0, str(PROJECT))

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set (finance_coordinator/.env)", file=sys.stderr)
        return 1
    if not os.getenv("OILPRICE_API_KEY"):
        print("Warning: OILPRICE_API_KEY not set — tools will return errors.", file=sys.stderr)

    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from energy_drilling_agent.agent import root_agent

    question = " ".join(sys.argv[1:]).strip() or "How many DUC wells are in the Permian right now?"
    runner = InMemoryRunner(agent=root_agent, app_name="drilling")
    await runner.session_service.create_session(app_name="drilling", user_id="u", session_id="s")
    msg = types.Content(role="user", parts=[types.Part(text=question)])

    final = None
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "function_call", None):
                    print(f"  [tool] {p.function_call.name}({dict(p.function_call.args)})", file=sys.stderr)
        if ev.is_final_response() and ev.content:
            final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))

    print("\n--- ENERGY DRILLING AGENT ---\n")
    print(final or "(no response)")
    from a2a_finance import observability
    observability.flush()  # send buffered Langfuse spans before exit
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
