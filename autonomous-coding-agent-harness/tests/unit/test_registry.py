from agent.retrieval.registry import build_registry, entry_text


class FakeTool:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.args = {"path": {"type": "string"}}


def test_build_registry_infers_git_namespace() -> None:
    entries = build_registry([FakeTool("git_status", "show status")])

    assert entries[0].namespace == "git"
    assert entries[0].name == "git_status"


def test_build_registry_defaults_to_fs_namespace() -> None:
    entries = build_registry([FakeTool("read_file", "read a file")])

    assert entries[0].namespace == "fs"


def test_entry_text_contains_tool_identity() -> None:
    entry = build_registry([FakeTool("read_file", "read a file")])[0]

    text = entry_text(entry)

    assert "read_file" in text
    assert "read a file" in text
