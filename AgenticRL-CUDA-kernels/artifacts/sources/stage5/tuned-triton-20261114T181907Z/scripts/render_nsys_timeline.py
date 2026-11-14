"""Turn an Nsight Systems SQLite export of utils/graph_trace_worker into timelines.

Torch-free and standard-library only. Reads the NVTX phase ranges written by the
worker, the CUDA runtime API calls and the GPU kernels/copies, assigns each GPU
activity to the phase whose host call launched it (by correlation ID), and
writes:

- a JSON summary per implementation and phase (counts, host and GPU spans);
- an SVG timeline per implementation: one column per phase, each drawn to its
  own time scale (labelled), with lanes for host API calls, GPU kernels and
  copies. Kernels that ran as CUDA-graph nodes are coloured separately.

Everything is measured from the trace; nothing is synthesized. Diagnostic only.
"""

import argparse
import html
import json
import re
import sqlite3
from pathlib import Path

PHASE = re.compile(r"^(?P<impl>[a-z_]+): (?P<phase>.+)$")
CAPTURE_CALLS = ("StreamBeginCapture", "StreamEndCapture", "GraphInstantiate", "ThreadExchangeStreamCaptureMode",
                 "StreamIsCapturing", "StreamGetCaptureInfo")
LAUNCH_CALLS = ("LaunchKernel", "LaunchCooperativeKernel", "cuLaunch", "LaunchHostFunc")
COPY_CALLS = ("Memcpy", "Memset")
SYNC_CALLS = ("Synchronize", "EventRecord", "EventQuery", "StreamWaitEvent", "EventSynchronize")
COLORS = {
    "graph_launch": "#c2410c", "capture": "#7c3aed", "launch": "#2563eb", "copy": "#0d9488",
    "sync": "#9ca3af", "other": "#d1d5db",
    "kernel_graph": "#c2410c", "kernel": "#2563eb", "memcpy_graph": "#f59e0b", "memcpy": "#0d9488",
}


def _columns(db, table):
    try:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def _strings(db):
    if not _columns(db, "StringIds"):
        return {}
    return dict(db.execute("SELECT id, value FROM StringIds"))


def api_category(name):
    if "GraphLaunch" in name:
        return "graph_launch"
    if any(token in name for token in CAPTURE_CALLS):
        return "capture"
    if any(token in name for token in LAUNCH_CALLS):
        return "launch"
    if any(token in name for token in COPY_CALLS):
        return "copy"
    if any(token in name for token in SYNC_CALLS):
        return "sync"
    return "other"


def load(path):
    """Read phases, host API calls and GPU activities from an nsys SQLite export."""
    db = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        strings = _strings(db)
        phases = []
        cols = _columns(db, "NVTX_EVENTS")
        if cols:
            text_expr = "text" if "text" in cols else "NULL"
            text_id = "textId" if "textId" in cols else "NULL"
            for start, end, text, tid in db.execute(
                    f"SELECT start, end, {text_expr}, {text_id} FROM NVTX_EVENTS "
                    "WHERE end IS NOT NULL AND end > start ORDER BY start"):
                label = text if text is not None else strings.get(tid)
                match = PHASE.match(label or "")
                if match:
                    phases.append({"impl": match["impl"], "phase": match["phase"], "start": start, "end": end})
        apis = []
        cols = _columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME")
        if cols:
            name_col = "nameId" if "nameId" in cols else "name"
            for start, end, corr, name in db.execute(
                    f"SELECT start, end, correlationId, {name_col} FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"):
                label = strings.get(name, name) if isinstance(name, int) else name
                label = str(label or "")
                apis.append({"start": start, "end": end, "corr": corr, "name": label,
                             "category": api_category(label)})
        gpu = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memcpy")):
            cols = _columns(db, table)
            if not cols:
                continue
            node = "graphNodeId" if "graphNodeId" in cols else "NULL"
            name = "shortName" if "shortName" in cols else "NULL"
            for start, end, corr, node_id, short in db.execute(
                    f"SELECT start, end, correlationId, {node}, {name} FROM {table} ORDER BY start"):
                label = strings.get(short, short) if isinstance(short, int) else short
                gpu.append({"start": start, "end": end, "corr": corr, "kind": kind,
                            "graph_node": bool(node_id), "name": str(label or kind)})
        return {"phases": phases, "apis": apis, "gpu": gpu}
    finally:
        db.close()


def assign(trace):
    """Attach host calls and GPU activities to the phase that issued them."""
    phases = sorted(trace["phases"], key=lambda p: p["start"])
    for phase in phases:
        phase["apis"], phase["gpu"] = [], []

    def owner(t):
        for phase in phases:
            if phase["start"] <= t <= phase["end"]:
                return phase
        return None

    by_corr = {}
    for call in trace["apis"]:
        phase = owner(call["start"])
        if phase is not None:
            phase["apis"].append(call)
            by_corr[call["corr"]] = phase
    for item in trace["gpu"]:
        phase = by_corr.get(item["corr"])
        if phase is not None:
            phase["gpu"].append(item)
    return phases


def summarize(phases):
    """Per implementation, per phase: counts and spans, all from the trace."""
    out = {}
    for phase in phases:
        kernels = [g for g in phase["gpu"] if g["kind"] == "kernel"]
        copies = [g for g in phase["gpu"] if g["kind"] == "memcpy"]
        gpu_items = phase["gpu"]
        cats = {}
        for call in phase["apis"]:
            cats[call["category"]] = cats.get(call["category"], 0) + 1
        out.setdefault(phase["impl"], []).append({
            "phase": phase["phase"],
            "host_us": (phase["end"] - phase["start"]) / 1000,
            "api_calls": len(phase["apis"]), "api_by_category": cats,
            "graph_launches": cats.get("graph_launch", 0),
            "kernels": len(kernels), "kernels_as_graph_nodes": sum(k["graph_node"] for k in kernels),
            "copies": len(copies), "copies_as_graph_nodes": sum(c["graph_node"] for c in copies),
            "gpu_span_us": ((max(g["end"] for g in gpu_items) - min(g["start"] for g in gpu_items)) / 1000
                            if gpu_items else None),
            "gpu_busy_us": sum(g["end"] - g["start"] for g in gpu_items) / 1000 if gpu_items else None,
            "distinct_kernels": sorted({k["name"] for k in kernels})[:12],
        })
    return out


def _fmt_us(us):
    if us is None:
        return "–"
    if us >= 1000:
        return f"{us / 1000:.3g} ms"
    return f"{us:.3g} µs"


def svg(impl, phases, title, *, column_width=168, max_columns=9):
    """One SVG: a column per phase, each on its own labelled time scale."""
    shown = [p for p in phases if p["impl"] == impl]
    if len(shown) > max_columns:  # keep the first and last replays visible
        shown = shown[:max_columns - 1] + shown[-1:]
    lanes = (("host API", "api"), ("GPU kernels", "kernel"), ("GPU copies", "memcpy"))
    left, top, lane_h, gap = 92, 74, 26, 10
    width = left + column_width * len(shown) + 16
    height = top + len(lanes) * (lane_h + gap) + 150
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="12" y="22" font-size="14" font-weight="600" fill="#111827">{html.escape(title)}</text>',
             '<text x="12" y="40" fill="#4b5563">Measured with Nsight Systems (--cuda-graph-trace=node). '
             'Each column is one NVTX phase on its own time scale; bar positions are to scale within the column.</text>']
    for li, (label, _) in enumerate(lanes):
        y = top + li * (lane_h + gap)
        parts.append(f'<text x="12" y="{y + 17}" fill="#374151">{label}</text>')
    for ci, phase in enumerate(shown):
        x0 = left + ci * column_width
        inner = column_width - 12
        t0, t1 = phase["start"], phase["end"]
        items = phase["gpu"]
        if items:
            t1 = max(t1, max(g["end"] for g in items))
        span = max(t1 - t0, 1)
        sx = lambda t: x0 + (t - t0) / span * inner  # noqa: E731
        kernels = sum(g["kind"] == "kernel" for g in items)
        nodes = sum(g["kind"] == "kernel" and g["graph_node"] for g in items)
        parts.append(f'<rect x="{x0}" y="{top - 26}" width="{inner}" height="{len(lanes) * (lane_h + gap) + 24}" '
                     'fill="#f9fafb" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 + 4}" y="{top - 12}" font-weight="600" fill="#111827">'
                     f'{html.escape(phase["phase"][:26])}</text>')
        for li, (_, kind) in enumerate(lanes):
            y = top + li * (lane_h + gap) + 4
            rows = phase["apis"] if kind == "api" else [g for g in items if g["kind"] == kind]
            for row in rows:
                if kind == "api":
                    color = COLORS[row["category"]]
                else:
                    color = COLORS[f"{kind}_graph" if row["graph_node"] else kind]
                x = sx(row["start"])
                w = max(sx(row["end"]) - x, 1.2)
                tip = html.escape(f'{row["name"]} {(row["end"] - row["start"]) / 1000:.3g} µs')
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{lane_h - 8}" fill="{color}">'
                             f'<title>{tip}</title></rect>')
        base = top + len(lanes) * (lane_h + gap) + 12
        stats = [f"window {_fmt_us(span / 1000)}", f"{len(phase['apis'])} host API calls",
                 f"{kernels} kernels ({nodes} as graph nodes)",
                 f"{sum(a['category'] == 'graph_launch' for a in phase['apis'])} cudaGraphLaunch"]
        for si, text in enumerate(stats):
            parts.append(f'<text x="{x0 + 4}" y="{base + si * 14}" fill="#374151">{html.escape(text)}</text>')
    legend = [("graph_launch", "cudaGraphLaunch"), ("capture", "capture / instantiate API"),
              ("launch", "kernel launch API"), ("copy", "memcpy/memset API"), ("sync", "sync / event API"),
              ("kernel_graph", "kernel run as graph node"), ("kernel", "kernel launched directly"),
              ("memcpy_graph", "copy as graph node"), ("memcpy", "copy launched directly")]
    ly = height - 58
    for i, (key, label) in enumerate(legend):
        lx = 12 + (i % 5) * 190
        yy = ly + (i // 5) * 18
        parts.append(f'<rect x="{lx}" y="{yy - 9}" width="10" height="10" fill="{COLORS[key]}"/>'
                     f'<text x="{lx + 14}" y="{yy}" fill="#374151">{html.escape(label)}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render(sqlite_path, out_dir, title):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = assign(load(sqlite_path))
    summary = summarize(phases)
    files = {}
    for impl in summary:
        path = out_dir / f"timeline-{impl}.svg"
        path.write_text(svg(impl, phases, f"{title} · {impl}"), encoding="utf-8")
        files[impl] = path.name
    (out_dir / "timeline-summary.json").write_text(
        json.dumps({"schema": "graph-trace-timeline-v1", "title": title, "svg": files,
                    "phases": summary}, indent=2) + "\n")
    return summary, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="CUDA-graph capture and replay")
    args = parser.parse_args(argv)
    summary, files = render(args.sqlite, args.out_dir, args.title)
    print(json.dumps({"svg": files, "impls": sorted(summary)}))


if __name__ == "__main__":
    main()
