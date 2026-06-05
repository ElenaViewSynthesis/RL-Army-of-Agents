from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph.context_manager import (
    build_ledger,
    compact_messages,
    estimate_tokens,
)


def _pair(index: int):
    ai = AIMessage(
        content="",
        id=f"ai-{index}",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": f"file_{index}.py"},
                "id": f"call-{index}",
                "type": "tool_call",
            }
        ],
    )
    tool = ToolMessage(
        content=f'{{"path": "file_{index}.py", "content": "data", "success": true}}',
        tool_call_id=f"call-{index}",
        id=f"tool-{index}",
    )
    return ai, tool


def test_estimate_tokens_counts_message_content() -> None:
    estimate = estimate_tokens([HumanMessage(content="x" * 40)])

    assert estimate == 10


def test_build_ledger_summarizes_tool_pairs() -> None:
    ai, tool = _pair(1)

    ledger = build_ledger([(ai, [tool])], "")

    assert "Progress ledger" in ledger
    assert "read_file" in ledger
    assert "file_1.py" in ledger


def test_compact_messages_preserves_recent_pairs_and_plan() -> None:
    messages = [HumanMessage(content="task")]
    for index in range(5):
        messages.extend(_pair(index))
    state = {
        "messages": messages,
        "progress_ledger": "",
        "ledger_message_id": None,
        "compaction_count": 0,
    }

    update = compact_messages(state)

    assert update["compaction_count"] == 1
    assert "Progress ledger" in update["progress_ledger"]
    assert update["token_estimate"] > 0
    assert update["messages"][-1].content.startswith("Progress ledger")
