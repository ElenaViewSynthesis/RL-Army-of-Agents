"""MCP server for the filesystem namespace."""

import re
import shutil
import sys
from itertools import islice
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.fs import (  # noqa: E402
    CopyInput,
    CopyOutput,
    DeleteInput,
    DeleteOutput,
    DirEntry,
    FileStatInput,
    FileStatOutput,
    GrepInput,
    GrepMatch,
    GrepOutput,
    ListDirInput,
    ListDirOutput,
    MakeDirInput,
    MakeDirOutput,
    MoveInput,
    MoveOutput,
    ReadFileInput,
    ReadFileOutput,
    ReadFileRangeInput,
    ReadFileRangeOutput,
    SearchFilesInput,
    SearchFilesOutput,
    WriteFileInput,
    WriteFileOutput,
)

mcp = FastMCP("fs")


def _error(model, **values) -> dict:
    return model(success=False, **values).model_dump()


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
        return _error(
            ReadFileOutput,
            path=request.path,
            content="",
            size_bytes=0,
            error=str(exc),
        )


@mcp.tool()
def read_file_range(
    path: str,
    start_line: int,
    end_line: int,
    encoding: str = "utf-8",
) -> dict:
    """Read a 1-indexed inclusive line range from a text file."""
    request = ReadFileRangeInput(
        path=path,
        start_line=start_line,
        end_line=end_line,
        encoding=encoding,
    )
    try:
        lines = Path(request.path).read_text(encoding=request.encoding).splitlines()
        return ReadFileRangeOutput(
            path=request.path,
            lines=lines[request.start_line - 1 : request.end_line],
            start_line=request.start_line,
            end_line=request.end_line,
        ).model_dump()
    except Exception as exc:
        return _error(
            ReadFileRangeOutput,
            path=request.path,
            lines=[],
            start_line=request.start_line,
            end_line=request.end_line,
            error=str(exc),
        )


@mcp.tool()
def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = False,
) -> dict:
    """Write text content to a file."""
    request = WriteFileInput(
        path=path,
        content=content,
        encoding=encoding,
        create_dirs=create_dirs,
    )
    try:
        target = Path(request.path)
        if request.create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(request.content, encoding=request.encoding)
        return WriteFileOutput(
            path=request.path,
            bytes_written=target.stat().st_size,
        ).model_dump()
    except Exception as exc:
        return _error(WriteFileOutput, path=request.path, bytes_written=0, error=str(exc))


@mcp.tool()
def list_dir(path: str = ".", recursive: bool = False) -> dict:
    """List directory entries."""
    request = ListDirInput(path=path, recursive=recursive)
    try:
        root = Path(request.path)
        iterator = root.rglob("*") if request.recursive else root.iterdir()
        entries = []
        for item in sorted(iterator, key=lambda p: str(p)):
            stat = item.stat()
            entries.append(
                DirEntry(
                    name=str(item.relative_to(root)) if request.recursive else item.name,
                    is_dir=item.is_dir(),
                    size_bytes=stat.st_size if item.is_file() else 0,
                )
            )
        return ListDirOutput(
            path=request.path,
            entries=entries,
            count=len(entries),
        ).model_dump()
    except Exception as exc:
        return _error(ListDirOutput, path=request.path, entries=[], count=0, error=str(exc))


@mcp.tool()
def search_files(root: str = ".", pattern: str = "*", max_results: int = 200) -> dict:
    """Search for files by glob pattern."""
    request = SearchFilesInput(root=root, pattern=pattern, max_results=max_results)
    try:
        matches = [
            str(path)
            for path in islice(Path(request.root).rglob(request.pattern), request.max_results)
            if path.is_file()
        ]
        return SearchFilesOutput(
            root=request.root,
            pattern=request.pattern,
            matches=matches,
            count=len(matches),
        ).model_dump()
    except Exception as exc:
        return _error(
            SearchFilesOutput,
            root=request.root,
            pattern=request.pattern,
            matches=[],
            count=0,
            error=str(exc),
        )


@mcp.tool()
def grep(
    root: str = ".",
    pattern: str = "",
    file_glob: str = "*",
    max_results: int = 200,
    case_sensitive: bool = True,
) -> dict:
    """Search file contents with a regular expression."""
    request = GrepInput(
        root=root,
        pattern=pattern,
        file_glob=file_glob,
        max_results=max_results,
        case_sensitive=case_sensitive,
    )
    try:
        flags = 0 if request.case_sensitive else re.IGNORECASE
        compiled = re.compile(request.pattern, flags)
        matches: list[GrepMatch] = []
        for file_path in Path(request.root).rglob(request.file_glob):
            if not file_path.is_file():
                continue
            for line_number, text in enumerate(
                file_path.read_text(errors="replace").splitlines(),
                1,
            ):
                if compiled.search(text):
                    matches.append(
                        GrepMatch(
                            file=str(file_path),
                            line_number=line_number,
                            text=text,
                        )
                    )
                    if len(matches) >= request.max_results:
                        break
            if len(matches) >= request.max_results:
                break
        return GrepOutput(
            root=request.root,
            pattern=request.pattern,
            matches=matches,
            count=len(matches),
        ).model_dump()
    except Exception as exc:
        return _error(
            GrepOutput,
            root=request.root,
            pattern=request.pattern,
            matches=[],
            count=0,
            error=str(exc),
        )


@mcp.tool()
def file_stat(path: str) -> dict:
    """Return file or directory metadata."""
    request = FileStatInput(path=path)
    try:
        target = Path(request.path)
        stat = target.stat()
        return FileStatOutput(
            path=request.path,
            size_bytes=stat.st_size,
            is_file=target.is_file(),
            is_dir=target.is_dir(),
        ).model_dump()
    except Exception as exc:
        return _error(
            FileStatOutput,
            path=request.path,
            size_bytes=0,
            is_file=False,
            is_dir=False,
            error=str(exc),
        )


@mcp.tool()
def make_dir(path: str, parents: bool = True, exist_ok: bool = True) -> dict:
    """Create a directory."""
    request = MakeDirInput(path=path, parents=parents, exist_ok=exist_ok)
    try:
        Path(request.path).mkdir(parents=request.parents, exist_ok=request.exist_ok)
        return MakeDirOutput(path=request.path).model_dump()
    except Exception as exc:
        return _error(MakeDirOutput, path=request.path, error=str(exc))


@mcp.tool()
def move(src: str, dst: str, overwrite: bool = False) -> dict:
    """Move or rename a file or directory."""
    request = MoveInput(src=src, dst=dst, overwrite=overwrite)
    try:
        destination = Path(request.dst)
        if destination.exists() and not request.overwrite:
            return _error(
                MoveOutput,
                src=request.src,
                dst=request.dst,
                error=f"destination exists: {request.dst}",
            )
        shutil.move(request.src, request.dst)
        return MoveOutput(src=request.src, dst=request.dst).model_dump()
    except Exception as exc:
        return _error(MoveOutput, src=request.src, dst=request.dst, error=str(exc))


@mcp.tool()
def delete(path: str, recursive: bool = False) -> dict:
    """Delete a file or, when recursive is true, a directory tree."""
    request = DeleteInput(path=path, recursive=recursive)
    try:
        target = Path(request.path)
        if target.is_dir():
            if request.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
        return DeleteOutput(path=request.path).model_dump()
    except Exception as exc:
        return _error(DeleteOutput, path=request.path, error=str(exc))


@mcp.tool()
def copy(src: str, dst: str, overwrite: bool = False) -> dict:
    """Copy a file or directory."""
    request = CopyInput(src=src, dst=dst, overwrite=overwrite)
    try:
        source = Path(request.src)
        destination = Path(request.dst)
        if destination.exists() and not request.overwrite:
            return _error(
                CopyOutput,
                src=request.src,
                dst=request.dst,
                error=f"destination exists: {request.dst}",
            )
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=request.overwrite)
        else:
            shutil.copy2(source, destination)
        return CopyOutput(src=request.src, dst=request.dst).model_dump()
    except Exception as exc:
        return _error(CopyOutput, src=request.src, dst=request.dst, error=str(exc))


if __name__ == "__main__":
    mcp.run()
