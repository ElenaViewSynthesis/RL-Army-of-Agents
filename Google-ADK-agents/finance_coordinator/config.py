"""Shared configuration for the finance coordinator agents.

Model selection is env-driven so the whole agent tree can be pointed at a
different Gemini model without touching code. Set ``ADK_MODEL`` in ``.env``.
"""

from __future__ import annotations

import os

# Default to Gemini 2.5 Flash — fast and cheap for tool-use + delegation.
# Override with e.g. ADK_MODEL=gemini-2.5-pro for deeper reasoning.
MODEL: str = os.getenv("ADK_MODEL", "gemini-2.5-flash")
