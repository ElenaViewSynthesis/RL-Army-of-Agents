#!/usr/bin/env python3
"""Seed the A2A store with commodities notes across coal / gas / oil / energy.

Each theme becomes a distinct document for semantic recall: its own `agent_runs`
envelope, `prices` rows from OilPrice tool fetches, and a narrative
`agent_responses` note. This script calls the tools directly rather than asking
an LLM to call them, so seeding is deterministic and does not depend on model
tool-calling quality.

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

THEME_CODES: dict[str, list[tuple[str, str, str]]] = {
    "coal": [
        ("NEWCASTLE_COAL_USD", "Newcastle coal (API6)", "past_month"),
    ],
    "gas": [
        ("NATURAL_GAS_GBP", "UK natural gas", "past_week"),
        ("NATURAL_GAS_TTF_SPOT_EUR", "TTF natural gas spot", "past_week"),
    ],
    "oil": [
        ("BRENT_CRUDE_USD", "Brent crude", "past_week"),
        ("WTI_USD", "WTI crude", "past_week"),
    ],
}


def _fmt_price(price: dict) -> str:
    if price.get("error"):
        return f"error: {price['error']}"
    if price.get("formatted"):
        return str(price["formatted"])
    value = price.get("price")
    currency = price.get("currency") or ""
    unit = price.get("unit") or ""
    if value is None:
        return "price unavailable"
    suffix = f" {currency}".rstrip()
    if unit:
        suffix += f" / {unit}"
    return f"{value}{suffix}"


def _history_read(history: dict) -> str:
    if history.get("error"):
        return f"history unavailable ({history['error']})"
    points = [
        p for p in history.get("prices", [])
        if isinstance(p.get("price"), (int, float))
    ]
    if len(points) < 2:
        return "not enough history for a trend read"
    first = float(points[0]["price"])
    last = float(points[-1]["price"])
    change = last - first
    direction = "up" if change > 0 else "down" if change < 0 else "flat"
    return (
        f"{direction} {change:+.2f} over the sampled period "
        f"(range {min(p['price'] for p in points):.2f}-{max(p['price'] for p in points):.2f})"
    )


def _code_note(theme: str, prompt: str) -> str:
    from commodities_agent.tools import get_commodity_history, get_commodity_price

    lines = [f"# {theme.upper()} commodity seed", "", prompt, ""]
    for code, label, period in THEME_CODES[theme]:
        price = get_commodity_price(code)
        history = get_commodity_history(code, period)
        lines.append(f"- {label} ({code}): {_fmt_price(price)}; {_history_read(history)}.")
    return "\n".join(lines)


def _energy_note(prompt: str) -> str:
    from commodities_agent.tools import list_fuse_watchlist

    data = list_fuse_watchlist()
    lines = ["# ENERGY commodity seed", "", prompt, ""]
    if data.get("error"):
        lines.append(f"- Fuse watchlist unavailable: {data['error']}")
        return "\n".join(lines)

    for theme, rows in (data.get("by_theme") or {}).items():
        lines.append(f"## {theme.title()}")
        for row in rows:
            label = row.get("label") or row.get("code")
            value = row.get("formatted")
            if not value:
                price = row.get("price")
                currency = row.get("currency") or ""
                unit = row.get("unit") or ""
                value = "price unavailable" if price is None else f"{price} {currency}".rstrip()
                if unit and value != "price unavailable":
                    value += f" / {unit}"
            if row.get("error"):
                value = f"error: {row['error']}"
            lines.append(f"- {label} ({row.get('code')}): {value}")
        lines.append("")
    lines.append("Overall read: seeded live prices across gas, petroleum, carbon, and coal.")
    return "\n".join(lines).strip()


async def run_one(theme: str, prompt: str) -> None:
    """Run one deterministic theme and persist run + prices + note."""
    from a2a_finance import storage

    run_id = storage.start_run("commodities_agent", subject=theme.upper(), prompt=prompt)
    print(f"\n=== [{theme}] run_id={run_id} ===", file=sys.stderr)

    final = _energy_note(prompt) if theme == "energy" else _code_note(theme, prompt)
    print(final[:400])
    storage.save_response(
        "commodities_agent", subject=theme.upper(), text=final, run_id=run_id
    )


async def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / "finance_coordinator" / ".env")
    sys.path.insert(0, str(PROJECT))

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
