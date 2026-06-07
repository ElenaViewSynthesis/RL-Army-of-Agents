"""Context tracking and deterministic compaction for long-horizon runs."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, SystemMessage, ToolMessage

from agent.graph.state import AgentState

COMPACT_THRESHOLD = int(os.environ.get("CONTEXT_COMPACT_THRESHOLD", "1200"))
KEEP_RECENT_PAIRS = int(os.environ.get("CONTEXT_KEEP_PAIRS", "3"))


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Estimate tokens by character count / 4."""
    total = 0
    for message in messages:
        content = message.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(block)) for block in content)
        else:
            total += len(str(content))
    return total // 4


def _extract_pairs(messages: list[BaseMessage]) -> list[tuple[AIMessage, list[ToolMessage]]]:
    """Group AI tool-call messages with following tool result messages."""
    pairs = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_messages = []
            cursor = index + 1
            while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
                tool_messages.append(messages[cursor])
                cursor += 1
            if tool_messages:
                pairs.append((message, tool_messages))
                index = cursor
                continue
        index += 1
    return pairs


def _compact_tool_content(raw: Any) -> str:
    """Extract a small deterministic summary from a tool result."""
    text = str(raw)
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text[:120]

    if isinstance(parsed, dict):
        if "path" in parsed:
            return f"path={parsed['path']} success={parsed.get('success', True)}"
        if "output" in parsed:
            return str(parsed["output"])[:120]
        if "files" in parsed:
            return f"files={len(parsed['files'])}"
        if "count" in parsed:
            return f"count={parsed['count']}"
    return text[:120]


def _summarize_pair(ai_message: AIMessage, tool_messages: list[ToolMessage]) -> str:
    tool_names = [call["name"] for call in ai_message.tool_calls or []]
    lines = []
    for index, tool_message in enumerate(tool_messages):
        name = tool_names[index] if index < len(tool_names) else "unknown"
        lines.append(f"- {name}: {_compact_tool_content(tool_message.content)}")
    return "\n".join(lines)


def build_ledger(
    pairs_to_compact: list[tuple[AIMessage, list[ToolMessage]]],
    prior_ledger: str,
) -> str:
    """Build a deterministic progress ledger from completed tool-call pairs."""
    entries = [_summarize_pair(ai, tools) for ai, tools in pairs_to_compact]
    parts = []
    if prior_ledger:
        parts.append(prior_ledger.removeprefix("Progress ledger:\n"))
    parts.extend(entries)
    return "Progress ledger:\n" + "\n".join(part for part in parts if part)


def compact_messages(state: AgentState) -> dict:
    """Compact old completed tool-call pairs into a ledger message."""
    messages = state.get("messages", [])
    pairs = _extract_pairs(messages)
    if len(pairs) <= KEEP_RECENT_PAIRS:
        return {"token_estimate": estimate_tokens(messages)}

    pairs_to_compact = pairs[:-KEEP_RECENT_PAIRS]
    ledger = build_ledger(pairs_to_compact, state.get("progress_ledger", ""))

    remove_ids = set()
    for ai_message, tool_messages in pairs_to_compact:
        if ai_message.id:
            remove_ids.add(ai_message.id)
        for tool_message in tool_messages:
            if tool_message.id:
                remove_ids.add(tool_message.id)

    prior_ledger_id = state.get("ledger_message_id")
    if prior_ledger_id:
        remove_ids.add(prior_ledger_id)

    ledger_message = SystemMessage(content=ledger, id=str(uuid.uuid4()))
    remaining = [message for message in messages if message.id not in remove_ids]

    return {
        "messages": [*(RemoveMessage(id=item) for item in remove_ids), ledger_message],
        "progress_ledger": ledger,
        "ledger_message_id": ledger_message.id,
        "token_estimate": estimate_tokens(remaining) + estimate_tokens([ledger_message]),
        "compaction_count": state.get("compaction_count", 0) + 1,
    }


def manage_context_node(state: AgentState) -> dict:
    """Update token estimate or compact when context grows past the threshold."""
    messages = state.get("messages", [])
    estimate = estimate_tokens(messages)
    if estimate < COMPACT_THRESHOLD:
        return {"token_estimate": estimate}
    return compact_messages(state)
