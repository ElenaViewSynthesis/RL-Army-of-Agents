"""Formatter agent — turns gathered analysis into a structured ResearchNote.

Has no tools (ADK forbids tools alongside ``output_schema``); it runs as the
second step of a two-step structured flow: the valuation agent gathers data,
this agent emits the schema-validated note. Runs on a non-reasoning OpenRouter
model so the JSON payload isn't polluted by a reasoning trace.
"""

from google.adk.agents import LlmAgent

from ..config import formatter_model
from ..schema import ResearchNote

note_formatter = LlmAgent(
    name="note_formatter",
    model=formatter_model(),
    description="Formats an analyst's notes into a structured ResearchNote.",
    instruction=(
        "You receive an equity analyst's working notes on a stock (including "
        "tool data and a verdict). Convert them into the structured research "
        "note. Fill every field from the notes; use null for price_target only "
        "if it is genuinely absent; rank risks most-severe first. Do not invent "
        "data that is not in the notes."
    ),
    output_schema=ResearchNote,
    output_key="research_note",
)
