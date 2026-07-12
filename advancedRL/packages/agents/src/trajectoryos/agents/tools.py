"""Coding tools executed inside a sandbox: read_file, edit_file, run_shell.

Tool results are environment observations — the episode loop records them with
``loss_mask=0`` always; they are never trainable tokens.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict
from trajectoryos.environments import Sandbox, SandboxError
from trajectoryos.schemas import ToolSpec

CODING_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="read_file",
        description="Read a file from the workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact, unique occurrence of old_string with new_string in a file. "
            "Creates the file when old_string is empty and the file does not exist."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolSpec(
        name="run_shell",
        description="Run a shell command in the workspace and return exit code and output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
            "required": ["command"],
        },
    ),
)


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    output: str
    modified_path: str | None = None
    modified_content: str | None = None


def execute_tool(
    sandbox: Sandbox, tool_name: str, arguments: dict[str, Any]
) -> ToolExecutionResult:
    try:
        if tool_name == "read_file":
            return ToolExecutionResult(ok=True, output=sandbox.read_file(str(arguments["path"])))
        if tool_name == "edit_file":
            return _edit_file(
                sandbox,
                path=str(arguments["path"]),
                old_string=str(arguments["old_string"]),
                new_string=str(arguments["new_string"]),
            )
        if tool_name == "run_shell":
            timeout = float(arguments.get("timeout_seconds", 120.0))
            result = sandbox.exec(str(arguments["command"]), timeout_seconds=timeout)
            output = (
                f"exit_code={result.exit_code} timed_out={result.timed_out}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            return ToolExecutionResult(ok=result.ok, output=output)
        return ToolExecutionResult(ok=False, output=f"unknown tool: {tool_name!r}")
    except (SandboxError, KeyError, TypeError, ValueError) as exc:
        return ToolExecutionResult(ok=False, output=f"{type(exc).__name__}: {exc}")


def _edit_file(
    sandbox: Sandbox, *, path: str, old_string: str, new_string: str
) -> ToolExecutionResult:
    if old_string == "":
        sandbox.write_file(path, new_string)
        return ToolExecutionResult(
            ok=True,
            output=f"created {path}",
            modified_path=path,
            modified_content=new_string,
        )
    content = sandbox.read_file(path)
    occurrences = content.count(old_string)
    if occurrences == 0:
        return ToolExecutionResult(ok=False, output=f"old_string not found in {path}")
    if occurrences > 1:
        return ToolExecutionResult(
            ok=False, output=f"old_string occurs {occurrences} times in {path}; must be unique"
        )
    updated = content.replace(old_string, new_string, 1)
    sandbox.write_file(path, updated)
    return ToolExecutionResult(
        ok=True,
        output=f"edited {path}",
        modified_path=path,
        modified_content=updated,
    )
