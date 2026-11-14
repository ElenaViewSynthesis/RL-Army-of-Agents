"""Write the two README figures for the tuned Stage 5 Triton kernels. Torch-free.

    python -m scripts.make_stage5_tuning_figures [--out assets/stage5]

fig_stage5_kv_reuse.svg     animated: dots are reads of one key/value tile; untuned has one program per
                            query row, tuned one per tile of 128 rows. CSS animation, so GitHub plays it
                            in a README <img>; prefers-reduced-motion shows a still frame with the totals.
fig_stage5_repeat_reads.svg dumbbell, log scale: repeated reads of the same data, untuned vs tuned.

The counts are design arithmetic from each kernel's launch grid at the catalog's acceptance case
(utils.task_catalog), not measurements. MAX_BLOCK_M mirrors the constant in
tasks/eagle_tree_*_attention/kernels/triton/*.py and BLOCK_V the one in candidate_topk.py.
"""

import argparse
import math
from pathlib import Path

from utils.task_catalog import select_cases

MAX_BLOCK_M = 128   # query rows per tuned attention program
UNTUNED, TUNED = "#eb6834", "#2a78d6"   # categorical slots 2 and 1 of the light palette
INK, MUTED, LINE, PANEL, BG = "#1f1e1c", "#55544f", "#c3c2b7", "#f3f2ee", "#ffffff"
FONT = "Helvetica, Arial, sans-serif"
CYCLE_S = 4.0


def counts():
    """Repeated reads per kernel at its acceptance case: (name, what, untuned, tuned, case id)."""
    rows = []
    for task, name in (("eagle_tree_verify_attention", "Tree verify attention"),
                       ("eagle_tree_row_attention", "Tree row attention")):
        case = select_cases(task, "acceptance")[0]
        p = case.params
        queries = p["ratio"] * p["n"]   # query rows per GQA group, contiguous in Q
        tiles = math.ceil(queries / min(MAX_BLOCK_M, max(16, 1 << (queries - 1).bit_length())))
        rows.append((name, "loads of each committed K/V tile per GQA group", queries, tiles, case.id))
    case = select_cases("eagle_candidate_topk", "acceptance")[0]
    rows.append(("Candidate top-k", "reads of the vocabulary weight matrix", case.params["b"], 1, case.id))
    return rows


def text(x, y, value, size=11, fill=MUTED, anchor="start", weight="normal", extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{extra}>{value}</text>')


def kv_reuse(queries, tiles):
    lanes_l = 16                         # drawn lanes; each stands for queries / 16 rows
    per_lane = queries // lanes_l
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 250" width="760" height="250" role="img" '
             f'aria-label="Animation: untuned, one program per query row, so the same key/value tile is read {queries} '
             f'times; tuned, one program per tile of 128 rows, so it is read {tiles} times.">']
    css = [f".dot{{offset-rotate:0deg;animation:move {CYCLE_S}s linear infinite}}",
           "@keyframes move{from{offset-distance:0%}to{offset-distance:100%}}",
           ".cnt{opacity:0}", ".final{opacity:0}",
           "@media (prefers-reduced-motion: reduce){.dot{animation:none;offset-distance:50%}.cnt{animation:none!important;opacity:0}.final{opacity:1}}"]
    body = [f'<rect width="760" height="250" fill="{BG}"/>',
            '<defs><marker id="h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>',
            text(10, 18, "Untuned: one program per query row", 12, INK, weight="bold"),
            text(400, 18, f"Tuned: one program per tile of {MAX_BLOCK_M} rows", 12, INK, weight="bold")]

    def counter(name, x, y, total, lanes, arrivals):
        # One text per count; each is visible from its arrival to the next (CSS steps, one cycle).
        out = []
        for k in range(lanes + 1):
            start = 0.0 if k == 0 else arrivals[k - 1]
            stop = 1.0 if k == lanes else arrivals[k]
            a, b = round(start * 100, 3), round(stop * 100, 3)
            frames = (f"0%{{opacity:0}}" if a > 0 else "") + f"{a}%{{opacity:1}}{b}%{{opacity:1}}" + \
                     (f"{round(b + 0.001, 3)}%{{opacity:0}}100%{{opacity:0}}" if b < 100 else "")
            css.append(f"@keyframes {name}{k}{{{frames}}}")
            css.append(f".{name}{k}{{animation:{name}{k} {CYCLE_S}s steps(1,end) infinite}}")
            value = round(k / lanes * total)
            out.append(text(x, y, f"{value:,} / {total:,} reads", 13, INK, weight="bold", extra=f' class="cnt {name}{k}"'))
        out.append(text(x, y, f"{total:,} / {total:,} reads", 13, INK, weight="bold", extra=' class="final"'))
        return out

    # Untuned side.
    phases = [(i + 0.5) / lanes_l for i in range(lanes_l)]
    for i in range(lanes_l):
        y = 32 + i * 9
        body.append(f'<rect x="10" y="{y}" width="110" height="7" rx="2" fill="{BG}" stroke="{LINE}"/>')
        body.append(f'<line x1="238" y1="86" x2="124" y2="{y + 3.5}" stroke="{LINE}" stroke-width="1" marker-end="url(#h)"/>')
        body.append(f'<circle class="dot" r="3.5" fill="{UNTUNED}" stroke="{BG}" stroke-width="1" '
                    f'style="offset-path:path(\'M238 86 L124 {y + 3.5}\');animation-delay:-{phases[i] * CYCLE_S:.3f}s"/>')
    body.append(text(10, 190, f"… {queries:,} query rows (8 heads × 64 nodes)"))
    body.append(f'<rect x="240" y="60" width="118" height="52" rx="4" fill="{PANEL}" stroke="{LINE}"/>')
    body.append(text(250, 80, "K/V tile", 12, INK, weight="bold"))
    body += counter("l", 250, 99, queries, lanes_l, sorted(1 - p for p in phases))
    # Tuned side.
    phases_r = [(i + 0.5) / tiles for i in range(tiles)]
    for i in range(tiles):
        y = 32 + i * 38
        body.append(f'<rect x="400" y="{y}" width="110" height="32" rx="3" fill="{BG}" stroke="{LINE}"/>')
        body.append(text(408, y + 20, f"rows {i * MAX_BLOCK_M}–{(i + 1) * MAX_BLOCK_M - 1}"))
        body.append(f'<line x1="628" y1="86" x2="514" y2="{y + 16}" stroke="{LINE}" stroke-width="1.5" marker-end="url(#h)"/>')
        body.append(f'<circle class="dot" r="4.5" fill="{TUNED}" stroke="{BG}" stroke-width="1" '
                    f'style="offset-path:path(\'M628 86 L514 {y + 16}\');animation-delay:-{phases_r[i] * CYCLE_S:.3f}s"/>')
    body.append(text(400, 190, f"each tile reused by its {MAX_BLOCK_M} rows from shared memory"))
    body.append(f'<rect x="630" y="60" width="120" height="52" rx="4" fill="{PANEL}" stroke="{LINE}"/>')
    body.append(text(640, 80, "K/V tile", 12, INK, weight="bold"))
    body += counter("r", 640, 99, tiles, tiles, sorted(1 - p for p in phases_r))
    body.append(text(10, 210, "Each moving dot is one read of the same key/value tile from device memory."))
    body.append(text(10, 225, "Counters: reads of this tile so far in one pass over the window; "
                              f"each untuned lane stands for {per_lane} rows."))
    body.append(text(10, 240, "Design arithmetic from the launch grid at the largest verify-attention case, not a measurement.", 10))
    parts.append("<style>" + "".join(css) + "</style>")
    parts += body
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def repeat_reads(rows):
    width, left, right, top, row_h = 640, 170, 70, 30, 40
    height = top + row_h * len(rows) + 40
    lo = -0.3   # axis starts just below 1x so the tuned value label clears the row names
    x = lambda v: left + ((math.log10(v) - lo) / (3 - lo)) * (width - left - right)   # log scale
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           'aria-label="Dumbbell chart, log scale: repeated reads of the same data, untuned against tuned, for tree verify '
           'attention, tree row attention and candidate top-k.">',
           f'<rect width="{width}" height="{height}" fill="{BG}"/>',
           f'<circle cx="{left}" cy="12" r="4.5" fill="{UNTUNED}"/>' + text(left + 9, 16, "Untuned", 10),
           f'<circle cx="{left + 70}" cy="12" r="4.5" fill="{TUNED}"/>' + text(left + 79, 16, "Tuned", 10)]
    for tick in (1, 10, 100, 1000):
        out.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 4}" y2="{top + row_h * len(rows)}" stroke="#e1e0d9" stroke-width="1"/>')
        out.append(text(f"{x(tick):.1f}", top + row_h * len(rows) + 16, f"{tick:,}×", 10, anchor="middle"))
    for i, (name, what, before, after, _) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(text(left - 12, y + 4, name, 12, INK, anchor="end"))
        out.append(f'<line x1="{x(after):.1f}" x2="{x(before):.1f}" y1="{y}" y2="{y}" stroke="{LINE}" stroke-width="2"/>')
        out.append(f'<circle cx="{x(before):.1f}" cy="{y}" r="5.5" fill="{UNTUNED}" stroke="{BG}" stroke-width="2"/>')
        out.append(f'<circle cx="{x(after):.1f}" cy="{y}" r="5.5" fill="{TUNED}" stroke="{BG}" stroke-width="2"/>')
        out.append(text(f"{x(before) + 10:.1f}", y + 4, f"{before:,}", 11, MUTED))
        out.append(text(f"{x(after) - 10:.1f}", y + 4, f"{after:,}", 11, MUTED, anchor="end"))
    out.append(text(10, height - 8, "Programs requesting the same data at the largest case; design arithmetic, not a measurement.", 10))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("assets/stage5"))
    args = parser.parse_args(argv)
    rows = counts()
    args.out.mkdir(parents=True, exist_ok=True)
    verify = rows[0]
    (args.out / "fig_stage5_kv_reuse.svg").write_text(kv_reuse(verify[2], verify[3]), encoding="utf-8")
    (args.out / "fig_stage5_repeat_reads.svg").write_text(repeat_reads(rows), encoding="utf-8")
    for name, what, before, after, case in rows:
        print(f"{name}: {before} -> {after} {what} ({case})")


if __name__ == "__main__":
    main()
