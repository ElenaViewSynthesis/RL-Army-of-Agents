"""Answer common questions about one Nsight Systems SQLite export (capture.sqlite).

Torch-free and standard-library only; needs neither nsys nor a GPU, so it runs on
any checkout against the capture.sqlite files the Modal runs already save.

    python -m scripts.query_nsys summary <capture.sqlite>
    python -m scripts.query_nsys kernels <capture.sqlite> --top 10 --sort total_ns
    python -m scripts.query_nsys nvtx    <capture.sqlite> [--domain 0]
    python -m scripts.query_nsys memcpy  <capture.sqlite>
    python -m scripts.query_nsys tree    <capture.sqlite> --query top_level
    python -m scripts.query_nsys tree    <capture.sqlite> --query most_time_consuming --root agenticrl_generated
    python -m scripts.query_nsys tree    <capture.sqlite> --query count_calls --event axpby_kernel
    python -m scripts.query_nsys tree    <capture.sqlite> --query subtree_summary --root agenticrl_generated

The result is one JSON object on the last stdout line (--pretty indents it).
All times are nanoseconds as recorded in the trace. A table the export does not
contain yields null or an empty list with reason "table_absent", never a zero
that looks measured. Diagnostic only: nothing here feeds the reward.

The tree puts host intervals (NVTX ranges, CUDA runtime API calls) inside each
other by time on their thread. GPU work (kernels, copies, synchronisations) hangs
under the host call that launched it, matched by correlation ID. Its time is GPU
time, which runs after the call returns, so it is not part of the call's duration.
"""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET = "CUPTI_ACTIVITY_KIND_MEMSET"
SYNC = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
NVTX = "NVTX_EVENTS"
EVENT_TYPES = ("kernel", "nvtx", "trace", "memcpy", "sync")
SORT_KEYS = ("total_ns", "count", "mean_ns", "max_ns")
QUERIES = ("top_level", "most_time_consuming", "count_calls", "subtree_summary")
PID_MASK = ~0xFFFFFF  # nsys: a globalTid with its low 24 bits cleared is its globalPid
DISTINCT_CAP = 8


def connect(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such SQLite export: {path}")
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _columns(db, table):
    try:
        return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
    except sqlite3.DatabaseError:
        return []


def _strings(db):
    if not _columns(db, "StringIds"):
        return {}
    return dict(db.execute("SELECT id, value FROM StringIds"))


def _rows(db, table, wanted):
    """Rows of `table` as dicts holding the `wanted` columns; a missing column reads as None."""
    cols = _columns(db, table)
    if not cols:
        return None
    exprs = ", ".join(c if c in cols else "NULL" for c in wanted)
    return [dict(zip(wanted, row)) for row in db.execute(f"SELECT {exprs} FROM {table} ORDER BY start")]


def _text(value, strings, fallback):
    if isinstance(value, int) and not isinstance(value, bool):
        value = strings.get(value)
    return str(value) if value not in (None, "") else fallback


def duration_stats(durations):
    """count, total, mean, population std, min, max and median of a non-empty list of ns durations."""
    return {"count": len(durations), "total_ns": sum(durations), "mean_ns": statistics.fmean(durations),
            "std_ns": statistics.pstdev(durations), "min_ns": min(durations), "max_ns": max(durations),
            "median_ns": statistics.median(durations)}


def _distinct(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return {"values": seen[:DISTINCT_CAP], "distinct": len(seen)}


# ---------------------------------------------------------------- flat reports

def event_summary(path):
    db = connect(path)
    try:
        counts, starts, ends = {}, [], []
        for label, table in (("kernel", KERNEL), ("memcpy", MEMCPY), ("memset", MEMSET), ("sync", SYNC),
                             ("runtime_api", RUNTIME), ("nvtx", NVTX)):
            if not _columns(db, table):
                counts[label] = None
                continue
            count, lo, hi = db.execute(f"SELECT COUNT(*), MIN(start), MAX(COALESCE(end, start)) FROM {table}").fetchone()
            counts[label] = count
            if count:
                starts.append(lo)
                ends.append(hi)
        span = {"start_ns": min(starts), "end_ns": max(ends), "duration_ns": max(ends) - min(starts)} if starts else None
        return {"report": "summary", "event_counts": counts, "time_span": span,
                "absent_tables": sorted(k for k, v in counts.items() if v is None)}
    finally:
        db.close()


def kernel_stats(path, top=None, sort="total_ns"):
    if sort not in SORT_KEYS:
        raise ValueError(f"sort must be one of {SORT_KEYS}")
    db = connect(path)
    try:
        strings = _strings(db)
        rows = _rows(db, KERNEL, ("start", "end", "shortName", "demangledName", "gridX", "gridY", "gridZ",
                                  "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory",
                                  "dynamicSharedMemory", "graphNodeId"))
    finally:
        db.close()
    if rows is None:
        return {"report": "kernels", "kernels": [], "launches": None, "total_ns": None, "reason": "table_absent"}
    groups = {}
    for row in rows:
        name = _text(row["shortName"], strings, _text(row["demangledName"], strings, "<unnamed>"))
        groups.setdefault(name, []).append(row)
    kernels = []
    for name, items in groups.items():
        entry = {"name": name, "demangled_name": _text(items[0]["demangledName"], strings, None)}
        entry.update(duration_stats([r["end"] - r["start"] for r in items]))
        entry["grid"] = _distinct([[r["gridX"], r["gridY"], r["gridZ"]] for r in items])
        entry["block"] = _distinct([[r["blockX"], r["blockY"], r["blockZ"]] for r in items])
        entry["registers_per_thread"] = _distinct([r["registersPerThread"] for r in items])
        entry["static_shared_memory"] = _distinct([r["staticSharedMemory"] for r in items])
        entry["dynamic_shared_memory"] = _distinct([r["dynamicSharedMemory"] for r in items])
        entry["graph_node_launches"] = sum(1 for r in items if r["graphNodeId"])
        kernels.append(entry)
    kernels.sort(key=lambda k: (-k[sort], k["name"]))
    return {"report": "kernels", "sort": sort, "distinct_kernels": len(kernels),
            "launches": len(rows), "total_ns": sum(k["total_ns"] for k in kernels),
            "kernels": kernels[:top] if top else kernels, "reason": None}


def nvtx_stats(path, domain=None):
    db = connect(path)
    try:
        strings = _strings(db)
        rows = _rows(db, NVTX, ("start", "end", "text", "textId", "domainId"))
    finally:
        db.close()
    if rows is None:
        return {"report": "nvtx", "ranges": [], "marks": None, "reason": "table_absent"}
    groups, marks = {}, 0
    for row in rows:
        if domain is not None and row["domainId"] != domain:
            continue
        if row["end"] is None:
            marks += 1
            continue
        text = row["text"] if row["text"] not in (None, "") else _text(row["textId"], strings, "<unnamed>")
        groups.setdefault((text, row["domainId"]), []).append(row["end"] - row["start"])
    ranges = [dict({"text": text, "domain_id": dom}, **duration_stats(durs)) for (text, dom), durs in groups.items()]
    ranges.sort(key=lambda r: (-r["total_ns"], r["text"]))
    return {"report": "nvtx", "domain": domain, "ranges": ranges, "marks": marks, "reason": None}


def memcpy_stats(path):
    db = connect(path)
    try:
        kinds = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")) if _columns(db, "ENUM_CUDA_MEMCPY_OPER") else {}
        copies = _rows(db, MEMCPY, ("start", "end", "bytes", "copyKind"))
        sets = _rows(db, MEMSET, ("start", "end", "bytes"))
    finally:
        db.close()
    if copies is None and sets is None:
        return {"report": "memcpy", "directions": [], "reason": "table_absent"}
    groups = {}
    for row in copies or []:
        kind = row["copyKind"]
        label = kinds.get(kind, f"copyKind {kind}" if kind is not None else "unknown direction")
        groups.setdefault(label, []).append(row)
    for row in sets or []:
        groups.setdefault("Memset", []).append(row)
    directions = []
    for label, items in groups.items():
        entry = dict({"direction": label}, **duration_stats([r["end"] - r["start"] for r in items]))
        sizes = [r["bytes"] for r in items]
        measured = all(size is not None for size in sizes)
        entry["total_bytes"] = sum(sizes) if measured else None
        entry["bytes_per_second"] = (entry["total_bytes"] / (entry["total_ns"] / 1e9)
                                     if measured and entry["total_ns"] > 0 else None)
        directions.append(entry)
    directions.sort(key=lambda d: (-d["total_ns"], d["direction"]))
    return {"report": "memcpy", "directions": directions, "reason": None}


# ---------------------------------------------------------------- interval tree

def _node(kind, name, start, end, tid=None):
    return {"type": kind, "name": name, "start": start, "end": end, "tid": tid, "children": []}


def build_tree(path, types=EVENT_TYPES, thread=None):
    """Return (root, info). Host intervals nest by time per thread; GPU work hangs under its launching call."""
    unknown = sorted(set(types) - set(EVENT_TYPES))
    if unknown:
        raise ValueError(f"unknown event types {unknown}; choose from {EVENT_TYPES}")
    db = connect(path)
    try:
        strings = _strings(db)
        nvtx = _rows(db, NVTX, ("start", "end", "text", "textId", "globalTid")) or []
        runtime = _rows(db, RUNTIME, ("start", "end", "nameId", "globalTid", "correlationId")) or []
        gpu = []
        for kind, table, name_col in (("kernel", KERNEL, "shortName"), ("memcpy", MEMCPY, None),
                                      ("memcpy", MEMSET, None), ("sync", SYNC, None)):
            if kind not in types:
                continue
            wanted = ("start", "end", "correlationId", "globalPid") + ((name_col,) if name_col else ())
            for row in _rows(db, table, wanted) or []:
                fallback = {"kernel": "<unnamed kernel>", "memcpy": "memset" if table == MEMSET else "memcpy",
                            "sync": "synchronization"}[kind]
                gpu.append((kind, _text(row.get(name_col), strings, fallback) if name_col else fallback, row))
    finally:
        db.close()

    calls = {}  # (globalPid or None, correlationId) -> launching runtime call
    host = []
    for row in runtime:
        if thread is not None and row["globalTid"] != thread:
            continue
        node = _node("trace", _text(row["nameId"], strings, "<unnamed call>"), row["start"], row["end"], row["globalTid"])
        pid = row["globalTid"] & PID_MASK if row["globalTid"] is not None else None
        calls[(pid, row["correlationId"])] = node
        if "trace" in types:
            host.append(node)
    if "nvtx" in types:
        for row in nvtx:
            if row["end"] is None or (thread is not None and row["globalTid"] != thread):
                continue
            text = row["text"] if row["text"] not in (None, "") else _text(row["textId"], strings, "<unnamed range>")
            host.append(_node("nvtx", text, row["start"], row["end"], row["globalTid"]))

    root = _node("root", "<capture>", None, None)
    by_thread = {}
    for node in host:
        by_thread.setdefault(node["tid"], []).append(node)
    for nodes in by_thread.values():
        nodes.sort(key=lambda n: (n["start"], -n["end"]))
        stack = []
        for node in nodes:
            while stack and not (stack[-1]["start"] <= node["start"] and node["end"] <= stack[-1]["end"]):
                stack.pop()
            (stack[-1] if stack else root)["children"].append(node)
            stack.append(node)

    def innermost(tid, moment):
        found, level = None, [n for n in root["children"] if n["tid"] == tid]
        while True:
            hit = next((n for n in level if n["type"] in ("nvtx", "trace")
                        and n["start"] <= moment <= n["end"]), None)
            if hit is None:
                return found
            found, level = hit, hit["children"]

    unattributed = 0
    for kind, name, row in gpu:
        node = _node(kind, name, row["start"], row["end"])
        call = calls.get((row["globalPid"], row["correlationId"]))
        if call is None:
            call = calls.get((None, row["correlationId"]))
        if call is None:
            if thread is not None:
                continue  # launched from another thread
            unattributed += 1
            root["children"].append(node)
            continue
        parent = call if "trace" in types else innermost(call["tid"], call["start"])
        (parent or root)["children"].append(node)
        if parent is None:
            unattributed += 1
    root["children"].sort(key=lambda n: n["start"])
    count = sum(1 for _ in _walk(root, None))
    return root, {"nodes": count, "top_level": len(root["children"]), "gpu_events_without_host_parent": unattributed,
                  "event_types": [t for t in EVENT_TYPES if t in types], "thread": thread}


def _walk(scope, max_depth):
    """Yield (node, depth, path) for every descendant of `scope` (depth 1 = its children)."""
    stack = [(child, 1, (child["name"],)) for child in reversed(scope["children"])]
    while stack:
        node, depth, path = stack.pop()
        yield node, depth, path
        if max_depth is None or depth < max_depth:
            stack.extend((child, depth + 1, path + (child["name"],)) for child in reversed(node["children"]))


def _matches(name, wanted, contains):
    return wanted in name if contains else name == wanted


def _brief(node):
    return {"name": node["name"], "type": node["type"], "start_ns": node["start"],
            "duration_ns": node["end"] - node["start"]}


def query_tree(path, query, event=None, root_name=None, max_depth=None, types=EVENT_TYPES, thread=None,
               contains=False, limit=50):
    if query not in QUERIES:
        raise ValueError(f"query must be one of {QUERIES}")
    if query == "count_calls" and not event:
        raise ValueError("count_calls needs --event NAME")
    if query == "subtree_summary" and not root_name:
        raise ValueError("subtree_summary needs --root NAME")
    root, info = build_tree(path, types, thread)
    if root_name:
        scopes = [node for node, _, _ in _walk(root, None) if _matches(node["name"], root_name, contains)]
    else:
        scopes = [root]
    result = {"report": "tree", "query": query, "root": root_name, "scopes_matched": len(scopes),
              "max_depth": max_depth, "tree": info}
    if not scopes:
        result["reason"] = "root_not_found"
        return result
    result["reason"] = None

    if query == "top_level":
        children = [child for scope in scopes for child in scope["children"]]
        result["total"] = len(children)
        result["intervals"] = [_brief(child) for child in children[:limit]]
        result["truncated"] = len(children) > limit
    elif query == "most_time_consuming":
        best = None
        for scope in scopes:
            for node, depth, path_names in _walk(scope, max_depth):
                if best is None or node["end"] - node["start"] > best[0]["end"] - best[0]["start"]:
                    best = (node, depth, path_names)
        result["event"] = (dict(_brief(best[0]), depth=best[1], path=list(best[2])) if best else None)
    elif query == "count_calls":
        durations = [node["end"] - node["start"] for scope in scopes for node, _, _ in _walk(scope, max_depth)
                     if _matches(node["name"], event, contains)]
        result["event"] = event
        result["count"] = len(durations)
        result["total_ns"] = sum(durations)
    else:  # subtree_summary
        by_type, by_name = {}, {}
        for scope in scopes:
            for node, _, _ in _walk(scope, max_depth):
                duration = node["end"] - node["start"]
                for table, key in ((by_type, node["type"]), (by_name, (node["type"], node["name"]))):
                    slot = table.setdefault(key, {"count": 0, "total_ns": 0})
                    slot["count"] += 1
                    slot["total_ns"] += duration
        result["own"] = duration_stats([scope["end"] - scope["start"] for scope in scopes])
        result["descendants_by_type"] = {key: by_type[key] for key in sorted(by_type)}
        ranked = sorted(by_name.items(), key=lambda item: (-item[1]["total_ns"], item[0]))
        result["descendants_by_name"] = [dict({"type": key[0], "name": key[1]}, **value) for key, value in ranked[:limit]]
        result["truncated"] = len(ranked) > limit
    return result


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretty", action="store_true", help="indent the JSON result")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, text):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("sqlite", type=Path)
        cmd.add_argument("--pretty", action="store_true", default=argparse.SUPPRESS, help="indent the JSON result")
        return cmd

    add("summary", "event counts per table and the capture's time span")
    kernels = add("kernels", "GPU kernel statistics grouped by kernel name")
    kernels.add_argument("--top", type=int, default=None)
    kernels.add_argument("--sort", choices=SORT_KEYS, default="total_ns")
    nvtx = add("nvtx", "NVTX range durations grouped by text")
    nvtx.add_argument("--domain", type=int, default=None)
    add("memcpy", "memory copies and sets grouped by direction")
    tree = add("tree", "structural queries over the interval tree")
    tree.add_argument("--query", choices=QUERIES, required=True)
    tree.add_argument("--event", help="event name for count_calls")
    tree.add_argument("--root", help="scope the query to subtrees with this name")
    tree.add_argument("--contains", action="store_true", help="match --event and --root as substrings")
    tree.add_argument("--max-depth", type=int, default=None)
    tree.add_argument("--types", default=",".join(EVENT_TYPES), help=f"comma-separated subset of {','.join(EVENT_TYPES)}")
    tree.add_argument("--thread", type=int, default=None, help="only this globalTid")
    tree.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            result = event_summary(args.sqlite)
        elif args.command == "kernels":
            result = kernel_stats(args.sqlite, args.top, args.sort)
        elif args.command == "nvtx":
            result = nvtx_stats(args.sqlite, args.domain)
        elif args.command == "memcpy":
            result = memcpy_stats(args.sqlite)
        else:
            result = query_tree(args.sqlite, args.query, args.event, args.root, args.max_depth,
                                tuple(t for t in args.types.split(",") if t), args.thread, args.contains, args.limit)
    except (FileNotFoundError, ValueError, sqlite3.DatabaseError) as exc:
        parser.exit(2, f"query_nsys: {exc}\n")
    result["sqlite"] = str(args.sqlite)
    print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
