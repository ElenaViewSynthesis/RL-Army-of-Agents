"""Render the Stage 7 honest-kernel timing chart as static SVG for the README (stdlib only).

Static version of the chart "Honest kernels at each task's largest bf16 case" on
the Stages 7 and 8 page of the Stage 6 Multi-LoRA Report artifact:

- fig_stage7_largest_case.svg: for each Stage 7 task and backend, the reward
  profiler's device time per forward for the generated kernel, eager PyTorch
  and torch.compile, on a log scale, with the reward reason earned.

Input is the committed suite rows of the acceptance run
(`artifacts/runs/<run>/suite/progress.jsonl`); nothing is measured here.

    python3 -m scripts.make_stage7_figures --out assets/stage7
"""

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
RUN = "stage7-tier1-acceptance-20261113T202705Z"
TASKS = ("relu_squared", "silu_and_mul", "geglu_tanh", "rotary_embedding")
BACKENDS = (("triton", "Triton"), ("cuda", "CUDA"), ("cutile", "cuTile"))
CASES = {"relu_squared": "4,096 × 14,336", "silu_and_mul": "4,096 × 2 × 14,336",
         "geglu_tanh": "4,096 × 2 × 16,384", "rotary_embedding": "8 × 8,192 tokens, D 128"}
# Same palette as the report artifact (light theme) and the Stage 6 figures.
GEN, EAGER, COMPILE = "#2a78d6", "#eb6834", "#1baf7a"
GOOD_TEXT, CRITICAL = "#006300", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def load_rows(run=RUN):
    path = ROOT / "artifacts" / "runs" / run / "suite" / "progress.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def honest_rows(rows):
    """One honest row per task and backend, in chart order; missing pairs are an error."""
    found = {(r["suite/task"], r["suite/backend"]): r for r in rows if r["suite/fixture"] == "honest"}
    return [(task, label, found[(task, backend)]) for task in TASKS for backend, label in BACKENDS]


def fmt_us(v):
    return f"{v / 1000:g} ms" if v >= 1000 else f"{v:g} µs"


def largest_case_svg(run=RUN):
    data = honest_rows(load_rows(run))
    values = [r[k] for _, _, r in data for k in ("env/t_gen_us", "env/t_eager_us", "env/t_compile_us")
              if r.get(k) is not None]
    lo, hi = 10 ** math.floor(math.log10(min(values))), 10 ** math.ceil(math.log10(max(values)))
    width, left, right, row_h, top = 900, 250, 150, 26, 92
    height = top + len(data) * row_h + 44
    span = width - left - right
    sx = lambda v: left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * span  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" {FONT} role="img" aria-label="Honest kernels at each Stage 7 task\'s largest bf16 case">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
           f'<text x="16" y="28" font-size="16" font-weight="600" fill="{INK}">Honest kernels at each task\'s largest bf16 case</text>',
           f'<text x="16" y="48" font-size="12" fill="{INK2}">Reward-profiler device time per forward, log scale · run {escape(run)} · one NVIDIA H200</text>']
    legend = ((GEN, "circle", "generated (authored kernel)"), (EAGER, "square", "eager PyTorch"),
              (COMPILE, "diamond", "torch.compile"))
    x = 16
    for color, shape, text in legend:
        out.append(marker(shape, x + 6, 70, color))
        out.append(f'<text x="{x + 16}" y="74" font-size="12" fill="{INK2}">{escape(text)}</text>')
        x += 32 + 7 * len(text)
    out.append(f'<text x="{width - right + 10}" y="{top - 8}" font-size="11" fill="{MUTED}">reward</text>')
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi:
        t = 10 ** e
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{height - 34}" stroke="{GRID}"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{height - 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{fmt_us(t)}</text>')
        e += 1
    for i, (task, backend, row) in enumerate(data):
        y = top + i * row_h + row_h / 2
        if i % len(BACKENDS) == 0:
            if i:
                out.append(f'<line x1="16" x2="{width - 16}" y1="{y - row_h / 2:.1f}" y2="{y - row_h / 2:.1f}" stroke="{AXIS}"/>')
            out.append(f'<text x="16" y="{y + 4:.1f}" font-size="12" font-family="ui-monospace, Consolas, monospace" fill="{INK}">{task}</text>')
            out.append(f'<text x="16" y="{y + 18:.1f}" font-size="10.5" fill="{MUTED}">{escape(CASES[task])} bf16</text>')
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11.5" fill="{INK2}" text-anchor="end">{backend}</text>')
        gen, eager, comp = (row.get(k) for k in ("env/t_gen_us", "env/t_eager_us", "env/t_compile_us"))
        points = [v for v in (gen, eager, comp) if v is not None]
        if len(points) > 1:
            out.append(f'<line x1="{sx(min(points)):.1f}" x2="{sx(max(points)):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{AXIS}" stroke-width="2"/>')
        tip = (f"{task} · {backend}: generated {gen:.1f} µs, eager {eager:.1f} µs, torch.compile {comp:.1f} µs, "
               f"{row.get('reward/reason')}") if None not in (gen, eager, comp) else f"{task} · {backend}: not measured"
        out.append(f'<g><title>{escape(tip)}</title>')
        if eager is not None:
            out.append(marker("square", sx(eager), y, EAGER))
        if comp is not None:
            out.append(marker("diamond", sx(comp), y, COMPILE))
        if gen is not None:
            out.append(marker("circle", sx(gen), y, GEN))
        out.append('</g>')
        reason = row.get("reward/reason") or "not measured"
        color, weight = ((GOOD_TEXT, ' font-weight="600"') if reason == "faster_than_both"
                         else (CRITICAL, "") if gen is None else (INK2, ""))
        out.append(f'<text x="{width - right + 10}" y="{y + 4:.1f}" font-size="11.5" fill="{color}"{weight}>'
                   f'{escape(reason.replace("_", " "))}</text>')
    out.append(f'<text x="16" y="{height - 4}" font-size="10.5" fill="{MUTED}">'
               f'artifacts/runs/{escape(run)}/suite/progress.jsonl → env/t_gen_us, env/t_eager_us, env/t_compile_us, reward/reason</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n", len(data)


def marker(shape, x, y, color):
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
    if shape == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>'
    return f'<path d="M{x:.1f} {y - 6:.1f} L{x + 6:.1f} {y:.1f} L{x:.1f} {y + 6:.1f} L{x - 6:.1f} {y:.1f} Z" fill="{color}"/>'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "stage7")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    svg, rows = largest_case_svg()
    (args.out / "fig_stage7_largest_case.svg").write_text(svg, encoding="utf-8")
    print(json.dumps({"fig_stage7_largest_case.svg": rows}))


if __name__ == "__main__":
    main()
