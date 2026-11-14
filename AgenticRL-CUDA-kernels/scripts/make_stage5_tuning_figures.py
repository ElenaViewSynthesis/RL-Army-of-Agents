"""Write the README figures for the tuned Stage 5 Triton kernels. Torch-free.

    python -m scripts.make_stage5_tuning_figures [--out assets/stage5]

fig_stage5_kv_reuse.svg     animated: dots are reads of one key/value tile; untuned has one program per
                            query row, tuned one per tile of 128 rows. CSS animation, so GitHub plays it
                            in a README <img>; prefers-reduced-motion shows a still frame with the totals.
fig_stage5_repeat_reads.svg dumbbell, log scale: repeated reads of the same data, untuned vs tuned.
fig_stage5_attention_pipeline.svg  animated: the five kernels of one tree-attention forward with their
                            launch grids at the largest verify-attention case, and what one program of
                            eagle_attention_committed does.

The counts are design arithmetic from each kernel's launch grid at the catalog's acceptance case
(utils.task_catalog), not measurements. MAX_BLOCK_M, BLOCK_N, MIN_PROGRAMS, MAX_SPLIT and MIN_SPLIT mirror
the constants in tasks/eagle_tree_*_attention/kernels/triton/*.py, and attention_plan() mirrors plan() there.
"""

import argparse
import math
from pathlib import Path

from utils.task_catalog import select_cases

MAX_BLOCK_M = 128   # query rows per tuned attention program
BLOCK_N = 64        # committed keys per inner step
MIN_PROGRAMS, MAX_SPLIT, MIN_SPLIT = 1024, 4096, 512
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


def attention_plan(task):
    """Launch grids of the five attention kernels at the task's acceptance case; mirrors the kernel's plan()."""
    case = select_cases(task, "acceptance")[0]
    p = case.params
    b, hk, n, hq = p["b"], p["hkv"], p["n"], p["hkv"] * p["ratio"]
    rows = (hq // hk) * n
    block_m = min(MAX_BLOCK_M, max(16, 1 << (rows - 1).bit_length()))
    tiles = math.ceil(rows / block_m)
    width = min(p["window"], p["s"])
    split = MAX_SPLIT
    while split > MIN_SPLIT and b * hk * tiles * math.ceil(width / split) < MIN_PROGRAMS:
        split //= 2
    committed = math.ceil(width / split)
    return {"case": case.id, "block_m": block_m, "split": split, "blocks": split // BLOCK_N, "splits": committed + 1,
            "ancestor": f"{b} × {n}", "visible": f"{b} × {n}", "committed": f"{b * hk} × {tiles} × {committed}",
            "tree": f"{b * hk} × {tiles}", "merge": f"{b * hq * n:,}"}


def attention_pipeline(plan):
    """Animated flow of one tree-attention forward; CSS only, so GitHub plays it in a README <img>."""
    width, height = 760, 392
    css = [".dot{offset-rotate:0deg;animation:flow linear infinite}",
           "@keyframes flow{from{offset-distance:0%}to{offset-distance:100%}}",
           "@media (prefers-reduced-motion: reduce){.dot{animation:none;offset-distance:50%}}"]
    body = [f'<rect width="{width}" height="{height}" fill="{BG}"/>',
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>']

    def node(x, y, w, title, sub=None):
        kernel = sub is not None
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="{6 if kernel else 14}" fill="{BG if kernel else PANEL}" '
               f'stroke="{MUTED if kernel else LINE}" stroke-width="1.5"/>',
               text(x + w / 2, y + (19 if kernel else 27), title, 12 if kernel else 11, INK if kernel else MUTED, "middle",
                    "bold" if kernel else "normal")]
        if kernel:
            out.append(text(x + w / 2, y + 35, sub + " programs", 11, MUTED, "middle"))
        return out

    for item in (node(10, 14, 80, "parents"), node(118, 14, 160, "eagle_ancestor_table", plan["ancestor"]),
                 node(306, 14, 70, "paths"), node(404, 14, 160, "eagle_visible_table", plan["visible"]),
                 node(592, 14, 80, "visible"),
                 node(10, 118, 80, "Q, K, V"), node(128, 118, 200, "eagle_attention_committed", plan["committed"]),
                 node(356, 118, 170, "split slots 0 … c−1"),
                 node(10, 212, 80, "Q, TK, TV"), node(128, 212, 200, "eagle_attention_tree", plan["tree"]),
                 node(356, 212, 170, "last split slot"),
                 node(560, 150, 180, "eagle_attention_merge", plan["merge"]), node(605, 236, 90, "Out")):
        body += item
    edges = [[(90, 37), (118, 37)], [(278, 37), (306, 37)], [(376, 37), (404, 37)], [(564, 37), (592, 37)],
             [(90, 141), (128, 141)], [(328, 141), (356, 141)], [(90, 235), (128, 235)], [(328, 235), (356, 235)],
             [(672, 37), (752, 37), (752, 290), (228, 290), (228, 258)],
             [(526, 141), (542, 141), (542, 164), (560, 164)], [(526, 235), (542, 235), (542, 184), (560, 184)],
             [(650, 196), (650, 236)]]
    dots = []
    for pts in edges:
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        points = " ".join(f"{x},{y}" for x, y in pts)
        body.append(f'<polyline points="{points}" fill="none" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#a)"/>')
        path = "M" + " L".join(f"{x} {y}" for x, y in pts)
        count, duration = max(1, round(length / 70)), length / 45
        for j in range(count):
            dots.append(f'<circle class="dot" r="3.5" fill="{TUNED}" stroke="{BG}" stroke-width="1.5" '
                        f"style=\"offset-path:path('{path}');animation-duration:{duration:.2f}s;"
                        f'animation-delay:-{duration * j / count:.2f}s"/>')
    body += dots
    body.append(f'<line x1="10" x2="{width - 10}" y1="310" y2="310" stroke="#e1e0d9" stroke-width="1"/>')
    body.append(text(10, 330, "eagle_attention_committed", 12, INK, weight="bold"))
    body.append(text(10, 347, f"The heavy kernel. One program takes a tile of {plan['block_m']} query rows of a GQA group and "
                              f"one split of {plan['split']:,} committed keys."))
    body.append(text(10, 362, f"It walks the split in {plan['blocks']} blocks of {BLOCK_N} keys: load the key and value block once, "
                              "score Q·Kᵀ on the tensor cores, update the"))
    body.append(text(10, 377, "running maximum and normaliser (online softmax), add P·V on the tensor cores, "
                              "then store its partial result in its split slot."))
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            'role="img" aria-label="Animated flow of one tree-attention forward: the ancestor table and visible mask, '
            'the committed and tree attention kernels writing split slots, and the merge kernel producing the output; '
            'with a description of eagle_attention_committed.">')
    return "\n".join([head, "<style>" + "".join(css) + "</style>", *body, "</svg>"]) + "\n"


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
    plan = attention_plan("eagle_tree_verify_attention")
    (args.out / "fig_stage5_attention_pipeline.svg").write_text(attention_pipeline(plan), encoding="utf-8")
    print("Attention pipeline: " + plan["committed"] + " committed programs, " + plan["case"])
    for name, what, before, after, case in rows:
        print(f"{name}: {before} -> {after} {what} ({case})")


if __name__ == "__main__":
    main()
