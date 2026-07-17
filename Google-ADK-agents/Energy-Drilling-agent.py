#!/usr/bin/env python3
"""Convenience launcher for the ADK-discoverable energy_drilling_agent package.

    uv run python Energy-Drilling-agent.py "How many DUC wells in the Permian?"

Equivalent to `uv run python -m energy_drilling_agent.run "..."`. The agent itself
lives in the `energy_drilling_agent/` package (so ADK's `adk web` / `adk run`
discover it); this flat-file shim just forwards to the package runner.
"""

import runpy

# Run the package's runner as __main__ (inherits sys.argv, so the question passes
# straight through).
runpy.run_module("energy_drilling_agent.run", run_name="__main__")
