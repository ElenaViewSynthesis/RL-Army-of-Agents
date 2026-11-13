"""Read-only, Torch-free tools over task-suite run evidence.

The logic behind the planned Modal results connector (see the local
HANDOFF_CODEX_MODAL_CONNECTOR.md): it reads run folders laid out as on the
`agenticrl-cuda-artifacts` volume and in `artifacts/runs/`:

    <root>/<run_id>/summary.json
    <root>/<run_id>/request.json
    <root>/<run_id>/suite/suite.json          (or <run_id>/suite.json)
    <root>/<run_id>/source/task-source-manifest.json  (or source-manifest.json)

Every function takes the root directory explicitly, never writes, validates
run IDs, stays inside the root, and returns only flat JSON scalars. Stage
logs (`log/*`), artifact paths (`artifact/*`), stdout/stderr, source files,
environment values and the GPU UUID are never returned. No MCP, Modal or web
dependency: a thin server wraps these functions.
"""

import json
import math
import re
from pathlib import Path

RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ROW_PREFIXES = ("suite/", "env/", "reward/", "task/", "bench/", "ceiling/", "meta/")
DROPPED_KEYS = frozenset(("bench/device_uuid", "ceiling/device_uuid"))
MAX_LIMIT = 500
MAX_TEXT = 512
HONEST_FIXTURES = ("honest", "chain_replay")
TIMING_COLUMNS = ("id", "task", "backend", "solution", "case", "result", "gen_us", "eager_us", "compile_us")


class ResultsError(ValueError):
    """A request the tools refuse: bad run id, missing run, or bad argument."""


def _scalar(value):
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_TEXT]
    return None


def _run_dir(root, run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ResultsError("run_id must be 1-128 letters, digits, '_' or '-'")
    base = Path(root).resolve()
    path = (base / run_id).resolve()
    if path.parent != base:
        raise ResultsError("run_id must name a folder directly under the results root")
    if not (path / "summary.json").is_file():
        raise ResultsError(f"no finished run named {run_id}")
    return path


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultsError(f"unreadable evidence file {Path(path).name}") from exc


def _first(*paths):
    return next((p for p in paths if p.is_file()), None)


def _suite(path):
    found = _first(path / "suite" / "suite.json", path / "suite.json")
    if found is None:
        raise ResultsError(f"{path.name} has no suite.json")
    return _load(found)


def _manifest(path):
    found = _first(path / "source" / "task-source-manifest.json", path / "source-manifest.json")
    return _load(found) if found else {}


def _limit(limit, default):
    limit = default if limit is None else limit
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ResultsError(f"limit must be an integer from 1 to {MAX_LIMIT}")
    return limit


def _summary_row(path):
    summary = _load(path / "summary.json")
    request = _load(path / "request.json") if (path / "request.json").is_file() else {}
    manifest = _manifest(path)
    counts = summary.get("counts") or {}
    tasks = request.get("tasks") or []
    return {key: _scalar(value) for key, value in {
        "run_id": path.name,
        "status": summary.get("status"),
        "passed": counts.get("passed"),
        "failed": counts.get("failed"),
        "unavailable": counts.get("unavailable"),
        "source_mismatches": len(summary.get("source_mismatches") or []),
        "wall_s": (summary.get("execution") or {}).get("wall_s"),
        "git_revision": manifest.get("git_revision"),
        "source_files": len(manifest.get("files") or {}) or None,
        "case_set": request.get("case_set"),
        "backends": request.get("backends"),
        "fixtures": request.get("fixtures"),
        "tasks": ",".join(tasks) if isinstance(tasks, list) else tasks,
        "gpu_requested": request.get("gpu_requested"),
    }.items()}


def list_suite_runs(root, prefix="task-", limit=None):
    """Finished runs (those with summary.json), newest run id first."""
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]{0,128}", prefix):
        raise ResultsError("prefix may only contain letters, digits, '_' or '-'")
    limit = _limit(limit, 50)
    base = Path(root)
    if not base.is_dir():
        return []
    names = sorted((p.name for p in base.iterdir()
                    if p.is_dir() and p.name.startswith(prefix) and RUN_ID.fullmatch(p.name)
                    and (p / "summary.json").is_file()), reverse=True)
    return [_summary_row(_run_dir(base, name)) for name in names[:limit]]


def get_suite_summary(root, run_id):
    """One row describing a finished run."""
    return [_summary_row(_run_dir(root, run_id))]


def _clean_row(row):
    return {key: _scalar(value) for key, value in row.items()
            if key.startswith(ROW_PREFIXES) and key not in DROPPED_KEYS}


def get_suite_rows(root, run_id, fixtures=None, tasks=None, limit=None, cursor=None):
    """A page of a run's suite rows: flat scalars only, log and artifact keys removed."""
    limit = _limit(limit, 200)
    cursor = 0 if cursor is None else cursor
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ResultsError("cursor must be a non-negative integer")
    for name, values in (("fixtures", fixtures), ("tasks", tasks)):
        if values is not None and (not isinstance(values, list) or not all(isinstance(v, str) for v in values)):
            raise ResultsError(f"{name} must be a list of strings")
    rows = [row for row in _suite(_run_dir(root, run_id)).get("rows", [])
            if (fixtures is None or row.get("suite/fixture") in fixtures)
            and (tasks is None or row.get("suite/task") in tasks)]
    page = [_clean_row(row) for row in rows[cursor:cursor + limit]]
    following = cursor + limit
    return {"rows": page, "total": len(rows), "next_cursor": following if following < len(rows) else None}


def get_honest_timings(root, run_id):
    """Honest and chain-replay rows in the dashboard CSV's columns (assets/stage5/*.csv)."""
    label = {"cuda": "CUDA", "triton": "Triton"}
    out = []
    for row in _suite(_run_dir(root, run_id)).get("rows", []):
        fixture = row.get("suite/fixture")
        if fixture not in HONEST_FIXTURES:
            continue
        backend = label.get(row.get("suite/backend"), row.get("suite/backend"))
        solution = "chain replay" if fixture == "chain_replay" else "honest"
        times = [row.get(key) for key in ("env/t_gen_us", "env/t_eager_us", "env/t_compile_us")]
        out.append({
            "id": f"{row.get('suite/task')} {backend}" + (" chain replay" if fixture == "chain_replay" else ""),
            "task": row.get("suite/task"), "backend": backend, "solution": solution,
            "case": row.get("suite/case"), "result": row.get("reward/reason"),
            **{name: (round(value, 3) if isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) else None)
               for name, value in zip(("gen_us", "eager_us", "compile_us"), times)},
        })
    return out


TOOLS = {
    "list_suite_runs": list_suite_runs,
    "get_suite_summary": get_suite_summary,
    "get_suite_rows": get_suite_rows,
    "get_honest_timings": get_honest_timings,
}
