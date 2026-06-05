"""MCP server for the initial filesystem namespace."""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.tool_io import ReadFileInput, ReadFileOutput  # noqa: E402

mcp = FastMCP("fs")


@mcp.tool()
def read_file(path: str, encoding: str = "utf-8") -> dict:
    """Read a text file and return structured contents."""
    request = ReadFileInput(path=path, encoding=encoding)
    try:
        raw = Path(request.path).read_bytes()
        return ReadFileOutput(
            path=request.path,
            content=raw.decode(request.encoding),
            size_bytes=len(raw),
        ).model_dump()
    except Exception as exc:
        return ReadFileOutput(
            path=request.path,
            content="",
            size_bytes=0,
            success=False,
            error=str(exc),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
