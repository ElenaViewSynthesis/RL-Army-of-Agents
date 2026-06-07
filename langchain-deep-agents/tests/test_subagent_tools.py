from app.tools.subagent_tools import extract_final_message


def test_extract_final_message_basic():
    fake = {"messages": [{"role": "assistant", "content": "final answer"}]}
    assert extract_final_message(fake) == "final answer"
