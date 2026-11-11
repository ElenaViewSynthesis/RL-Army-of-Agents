"""Render the Stage 5 largest-case figures and table for publication (stdlib only).

Reads the honest-solution timings CSV (one row per task/backend/solution, from
H200 run task-stage5-gpu-acceptance-20261110T002043Z) and writes vector SVG
figures plus LaTeX and Markdown tables. No plotting library is required.

    python3 -m scripts.make_stage5_figures \\
        --csv assets/stage5/stage5-largest-honest.csv --out assets/stage5

Convert an SVG to PDF for LaTeX with, for example, `rsvg-convert -f pdf`.
Colors are the validated categorical slots 1-2 (blue, orange) with shape as a
second cue (square = CUDA, triangle = Triton, hollow = chain replay); eager and
torch.compile are neutral context marks.
"""

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

LABELS = {
    "eagle_candidate_topk": "Candidate top-k",
    "eagle_tree_step": "Tree step",
    "eagle_tree_row_attention": "Row attention (32k)",
    "eagle_tree_verify_attention": "Verify attention (32k)",
    "eagle_gdn_tree_conv": "GDN tree conv",
    "eagle_mamba2_tree_conv": "Mamba2 tree conv",
    "eagle_gdn_tree_recurrence": "GDN recurrence",
    "eagle_accept_tail": "Greedy acceptance",
    "eagle_accept_tail_sampled": "Sampled acceptance",
}
CUDA, TRITON = "#2a78d6", "#eb6834"
INK, MUTED, GRID, REF, PAPER = "#1d1d1b", "#55544f", "#dedcd6", "#8f8d86", "#ffffff"
FONT = "Helvetica, Arial, sans-serif"
CHAR_W = 0.56  # average glyph width in em, for label margins


def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("gen_us", "eager_us", "compile_us"):
            row[key] = float(row[key])
        row["speedup"] = row["eager_us"] / row["gen_us"]
    return rows


def tasks_in_order(rows):
    return list(dict.fromkeys(row["task"] for row in rows))


def log_axis(lo, hi, x0, x1):
    lo_e, hi_e = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    def x(value):
        return x0 + (math.log10(value) - lo_e) / (hi_e - lo_e) * (x1 - x0)
    return x, [10.0 ** e for e in range(lo_e, hi_e + 1)]


def marker(kind, cx, cy, color, hollow=False, size=4.6):
    fill = PAPER if hollow else color
    stroke = color if hollow else PAPER
    if kind == "square":
        s = size * 0.9
        return (f'<rect x="{cx - s:.2f}" y="{cy - s:.2f}" width="{2 * s:.2f}" height="{2 * s:.2f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    if kind == "triangle":
        s = size * 1.15
        pts = f"{cx:.2f},{cy - s:.2f} {cx + s:.2f},{cy + s * 0.8:.2f} {cx - s:.2f},{cy + s * 0.8:.2f}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
    return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.4"/>')


def text(x, y, value, size=9, anchor="start", color=INK, weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')


def frame(width, height, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{escape(label)}">\n'
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>\n' + "\n".join(body) + "\n</svg>\n")


def series_of(row):
    if row["solution"] == "chain replay":
        return "triangle", TRITON, True, "Triton chain replay"
    if row["backend"] == "CUDA":
        return "square", CUDA, False, "CUDA"
    return "triangle", TRITON, False, "Triton"


def legend(entries, x, y):
    out, cx = [], x
    for kind, color, hollow, name in entries:
        out.append(marker(kind, cx + 5, y - 3, color, hollow))
        out.append(text(cx + 14, y, name, 8.5, color=MUTED))
        cx += 22 + len(name) * 8.5 * CHAR_W + 12
    return out


def label_width(rows, size=9):
    return max(len(LABELS.get(t, t)) for t in tasks_in_order(rows)) * size * CHAR_W


def speedup_figure(rows):
    """Figure 1: eager time / solution time, log scale, one row per task."""
    tasks = tasks_in_order(rows)
    width, row_h, top, bottom = 520, 26, 46, 40
    left = label_width(rows) + 18
    right = 18
    height = top + bottom + row_h * len(tasks)
    lo = min(r["speedup"] for r in rows) / 1.4
    hi = max(r["speedup"] for r in rows) * 1.4
    x, decades = log_axis(lo, hi, left, width - right)
    y = {t: top + row_h * (i + 0.5) for i, t in enumerate(tasks)}
    body = legend([("square", CUDA, False, "CUDA"), ("triangle", TRITON, False, "Triton"),
                   ("triangle", TRITON, True, "Triton chain replay")], left, 16)
    for d in decades:
        stroke, w = (REF, 1.4) if d == 1 else (GRID, 0.7)
        body.append(f'<line x1="{x(d):.2f}" x2="{x(d):.2f}" y1="{top - 6}" y2="{height - bottom + 2}" '
                    f'stroke="{stroke}" stroke-width="{w}"/>')
        label = f"{d:g}×" if d < 1 else f"{int(d)}×"
        body.append(text(x(d), height - bottom + 15, label, 8, "middle", MUTED))
    body.append(text(x(1) - 5, top - 10, "← slower than eager", 8, "end", MUTED))
    body.append(text(x(1) + 5, top - 10, "faster than eager →", 8, "start", MUTED))
    body.append(text((left + width - right) / 2, height - 6, "Eager time ÷ solution time (log scale)", 9, "middle", INK))
    for t in tasks:
        body.append(f'<line x1="{left}" x2="{width - right}" y1="{y[t]:.2f}" y2="{y[t]:.2f}" '
                    f'stroke="{GRID}" stroke-width="0.6" stroke-dasharray="2 3"/>')
        body.append(text(left - 8, y[t] + 3.2, LABELS.get(t, t), 9, "end"))
    for r in rows:
        kind, color, hollow, _ = series_of(r)
        dy = 6 if hollow else (-4 if r["backend"] == "CUDA" else 4)
        body.append(marker(kind, x(r["speedup"]), y[r["task"]] + dy, color, hollow))
    return frame(width, height, body,
                 "Stage 5 honest solutions: speed relative to eager PyTorch at the largest case, log scale.")


def time_figure(rows):
    """Figure 2: time per call for eager, torch.compile, CUDA and Triton, log scale."""
    tasks = tasks_in_order(rows)
    width, row_h, top, bottom = 520, 34, 46, 40
    left = label_width(rows) + 18
    right = 18
    height = top + bottom + row_h * len(tasks)
    values = [v for r in rows for v in (r["gen_us"], r["eager_us"], r["compile_us"])]
    x, decades = log_axis(min(values) / 1.3, max(values) * 1.3, left, width - right)
    y = {t: top + row_h * (i + 0.5) for i, t in enumerate(tasks)}
    body = legend([("circle", MUTED, False, "Eager"), ("circle", MUTED, True, "torch.compile"),
                   ("square", CUDA, False, "CUDA"), ("triangle", TRITON, False, "Triton"),
                   ("triangle", TRITON, True, "chain replay")], left, 16)
    unit = {0: "1 µs", 1: "10 µs", 2: "100 µs", 3: "1 ms", 4: "10 ms", 5: "100 ms", 6: "1 s", 7: "10 s"}
    for d in decades:
        e = round(math.log10(d))
        body.append(f'<line x1="{x(d):.2f}" x2="{x(d):.2f}" y1="{top - 6}" y2="{height - bottom + 2}" '
                    f'stroke="{GRID}" stroke-width="0.7"/>')
        body.append(text(x(d), height - bottom + 15, unit.get(e, f"1e{e} µs"), 8, "middle", MUTED))
    body.append(text((left + width - right) / 2, height - 6,
                     "Time per call, profiler device time (log scale)", 9, "middle", INK))
    for t in tasks:
        group = [r for r in rows if r["task"] == t]
        eager = sum(r["eager_us"] for r in group) / len(group)
        comp = sum(r["compile_us"] for r in group) / len(group)
        xs = [x(eager), x(comp)] + [x(r["gen_us"]) for r in group]
        body.append(f'<line x1="{min(xs):.2f}" x2="{max(xs):.2f}" y1="{y[t]:.2f}" y2="{y[t]:.2f}" '
                    f'stroke="{GRID}" stroke-width="2"/>')
        body.append(text(left - 8, y[t] + 3.2, LABELS.get(t, t), 9, "end"))
        # Compile first, eager smaller on top, so equal times stay visible.
        body.append(marker("circle", x(comp), y[t], MUTED, hollow=True))
        body.append(marker("circle", x(eager), y[t], MUTED, size=3.4))
        for r in group:
            kind, color, hollow, _ = series_of(r)
            dy = 10 if hollow else (-8 if r["backend"] == "CUDA" else 8)
            body.append(marker(kind, x(r["gen_us"]), y[t] + dy, color, hollow))
    return frame(width, height, body,
                 "Stage 5 time per call at the largest case for eager, torch.compile, CUDA and Triton, log scale.")


def fmt_us(value):
    return f"{value:,.1f}"


def ratio(value):
    return f"{value:.1f}× faster" if value >= 1 else f"{1 / value:.1f}× slower"


def latex_table(rows):
    def tex(s):
        return s.replace("_", r"\_").replace("×", r"$\times$")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Stage 5 (EAGLE-4) honest solutions at each task's largest case on one H200: "
             r"profiler device time per call in \si{\micro\second} and speed relative to eager PyTorch. "
             r"All 71 suite rows passed (run task-stage5-gpu-acceptance-20261110T002043Z).}",
             r"\label{tab:stage5-largest}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Task & Solution & Solution & Eager & \texttt{torch.compile} & vs eager \\",
             r"\midrule"]
    for r in rows:
        solution = r["backend"] + (" chain replay" if r["solution"] == "chain replay" else "")
        lines.append(" & ".join([tex(LABELS.get(r["task"], r["task"])), tex(solution), fmt_us(r["gen_us"]),
                                 fmt_us(r["eager_us"]), fmt_us(r["compile_us"]), tex(ratio(r["speedup"]))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def markdown_table(rows):
    lines = ["| Task | Solution | Reward reason | Solution µs | Eager µs | torch.compile µs | vs eager |",
             "|---|---|---|---:|---:|---:|---|"]
    for r in rows:
        solution = r["backend"] + (" chain replay" if r["solution"] == "chain replay" else "")
        lines.append(f"| {LABELS.get(r['task'], r['task'])} | {solution} | {r['result']} | {fmt_us(r['gen_us'])} | "
                     f"{fmt_us(r['eager_us'])} | {fmt_us(r['compile_us'])} | {ratio(r['speedup'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("assets/stage5/stage5-largest-honest.csv"))
    parser.add_argument("--out", type=Path, default=Path("assets/stage5"))
    args = parser.parse_args(argv)
    rows = load(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"fig_stage5_speedup_vs_eager.svg": speedup_figure(rows),
               "fig_stage5_time_per_call.svg": time_figure(rows),
               "table_stage5_largest.tex": latex_table(rows),
               "table_stage5_largest.md": markdown_table(rows)}
    for name, content in outputs.items():
        (args.out / name).write_text(content, encoding="utf-8", newline="\n")
        print(args.out / name)


if __name__ == "__main__":
    main()
