"""Token-preserving tool policy backed by an SGLang engine.

Owns the exact token context fed to the engine:

- Prompt template and tool observations are tokenized once, reported to the
  episode loop via ``PolicyTurn.context_token_ids`` (recorded with
  ``loss_mask=0``).
- Sampled completions come back from SGLang as token IDs + logprobs and are
  reported verbatim with ``loss_mask=1``. Text is decoded only to parse tool
  calls; it is never re-tokenized into the training stream.

Tool-call protocol (plain-text, template-agnostic):

    <tool_call>{"tool_name": "...", "arguments": {...}}</tool_call>
"""

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from trajectoryos.agents import PolicyTurn, ToolCall
from trajectoryos.integrations.sglang.client import SGLangClient
from trajectoryos.schemas import EventType, TaskSpec, TrajectoryEvent

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

DEFAULT_SYSTEM_TEMPLATE = """You are a coding agent working in a repository sandbox.
Fix the described issue using the available tools, then verify by running the tests.

Available tools:
{tool_lines}

To call a tool respond with exactly:
<tool_call>{{"tool_name": "<name>", "arguments": {{...}}}}</tool_call>

When the issue is fixed and tests pass, respond without any tool call to submit.
"""


class Tokenizer(Protocol):
    """Minimal tokenizer surface; satisfied by Hugging Face tokenizers."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract the first well-formed tool call; None means the policy submits."""
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
        return ToolCall(tool_name=str(data["tool_name"]), arguments=dict(data["arguments"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def render_system_prompt(task: TaskSpec) -> str:
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in task.tools) or "- (none)"
    return DEFAULT_SYSTEM_TEMPLATE.format(tool_lines=tool_lines)


class SGLangToolPolicy:
    def __init__(
        self,
        client: SGLangClient,
        tokenizer: Tokenizer,
        task: TaskSpec,
        *,
        sampling_params: dict[str, Any] | None = None,
        max_context_tokens: int = 32768,
    ) -> None:
        self._client = client
        self._tokenizer = tokenizer
        self._task = task
        self._sampling_params = sampling_params or {"temperature": 0.7, "max_new_tokens": 1024}
        self._max_context_tokens = max_context_tokens
        self._context_ids: list[int] = []
        self._started = False
        self._last_seen_event = 0

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _pending_context(self, events: Sequence[TrajectoryEvent]) -> list[int]:
        """Tokenize everything new since the previous turn, exactly once."""
        chunks: list[str] = []
        if not self._started:
            chunks.append(render_system_prompt(self._task))
            chunks.append(f"\nIssue:\n{self._task.prompt}\n")
            self._started = True
        for event in events[self._last_seen_event :]:
            if event.event_type is EventType.TOOL_RESULT:
                chunks.append(f"\n<tool_result>\n{event.tool_result}\n</tool_result>\n")
        self._last_seen_event = len(events)
        return self._encode("".join(chunks)) if chunks else []

    def next_turn(self, events: Sequence[TrajectoryEvent]) -> PolicyTurn:
        context_ids = self._pending_context(events)
        self._context_ids.extend(context_ids)
        if len(self._context_ids) > self._max_context_tokens:
            # Submit rather than overflow; the budget tracker records the cost.
            return PolicyTurn(
                content="(context limit reached; submitting)",
                context_token_ids=context_ids,
                input_tokens_used=len(context_ids),
            )

        output = self._client.generate(list(self._context_ids), self._sampling_params)
        # The sampled tokens become part of the next turn's context, verbatim.
        self._context_ids.extend(output.token_ids)

        return PolicyTurn(
            content=output.text,
            tool_call=parse_tool_call(output.text),
            token_ids=list(output.token_ids),
            rollout_logprobs=list(output.logprobs),
            loss_mask=[1] * len(output.token_ids),
            context_token_ids=context_ids or None,
            input_tokens_used=len(context_ids),
            output_tokens_used=len(output.token_ids),
        )
