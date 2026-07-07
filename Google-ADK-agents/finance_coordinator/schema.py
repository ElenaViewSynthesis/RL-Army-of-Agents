"""Structured research-note schema (Pydantic) — the Python analogue of the TS
zod schema. Used as an ADK ``output_schema`` on a formatter agent to produce a
schema-validated note.

Note: an ADK ``LlmAgent`` with ``output_schema`` set cannot also call tools, so
structured output is a two-step design — a tool-using agent gathers the data,
then a tool-less formatter agent emits this schema.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Rating(str, Enum):
    buy = "BUY"
    hold = "HOLD"
    sell = "SELL"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ResearchNote(BaseModel):
    """A full equity research note."""

    ticker: str = Field(description="Ticker analyzed, e.g. NVDA")
    rating: Rating
    current_price: float | None = Field(
        default=None, description="Latest price, or null if unavailable"
    )
    price_target: float | None = Field(
        default=None, description="12-month price target, or null if data is insufficient"
    )
    fundamentals: str = Field(
        description="2-4 sentences: business, profitability, balance-sheet strength"
    )
    valuation: str = Field(
        description="DCF fair value vs price, peer context, analyst consensus"
    )
    risks: list[str] = Field(description="Ranked list of the top material risks")
    confidence: Confidence
    summary: str = Field(description="Executive summary tying the sections together")
