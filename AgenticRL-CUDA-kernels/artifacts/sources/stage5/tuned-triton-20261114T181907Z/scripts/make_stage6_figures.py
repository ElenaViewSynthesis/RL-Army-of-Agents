"""Render two Stage 6 figures as static SVG for the README (stdlib only).

Static versions of two charts in the Stage 6 Multi-LoRA Report artifact:

- fig_stage6_slot_ranks.svg: per-slot ranks for every Stage 6 case in the
  evidence, with the padding to a multiple of 16 hatched (where wrong_padding
  has something to catch).
- fig_stage6_coverage_stratified.svg: the coverage matrix of the stratified run,
  one cell per task x case x backend x fixture, with its legend.

Inputs are the committed suite rows (`artifacts/runs/<run>/suite/progress.jsonl`)
and the Torch-free case contracts in utils/task_catalog.py and
utils/lora_contracts.py; nothing is measured here.

    python3 -m scripts.make_stage6_figures --out assets/stage6
"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from utils.lora_contracts import LORA_TASKS, slot_ranks
from utils.task_catalog import get_case

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("stage6-smoke-20261110T105217Z", "stage6-smoke-rerun-20261111T031847Z",
        "stage6-acceptance-20261112T030910Z", "stage6-stratified-20261112T094226Z")
COVERAGE_RUN = "stage6-stratified-20261112T094226Z"
FIXTURES = ("honest", "hack", "wrong", "wrong_padding", "wrong_skip")
BACKENDS = ("cuda", "triton")
TIER = {"faster_than_both": 3, "faster_than_eager": 2, "correct": 1}
# Validated reference palette (dataviz skill): ordinal blue ramp, status colours.
SEQ = ("#86b6ef", "#3987e5", "#184f95")
GOOD, CRITICAL, WARN = "#0ca30c", "#d03b3b", "#fab219"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def label(task, case_id):
    """Compact label: t<tokens> h<hidden> r<rank> s<slots>×<slices> <pattern> [mixed] <dtype>."""
    item = get_case(task, case_id)
    p = dict(item.params)
    dtype = {"float16": "fp16", "bfloat16": "bf16"}.get(item.dtype, item.dtype)
    return (f"t{p['tokens']} h{p['hidden']} r{p['rank']} s{p['slots']}×{p['slices']} {p['pattern']}"
            f"{' mixed' if p['mixed'] else ''} {dtype}")


def load_rows(run):
    path = ROOT / "artifacts" / "runs" / run / "suite" / "progress.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ramp(i, n):
    """Interpolate the three ordinal steps by slot index."""
    def rgb(h):
        return [int(h[k:k + 2], 16) for k in (1, 3, 5)]
    a, b, c = (rgb(h) for h in SEQ)
    t = 0.5 if n <= 1 else i / (n - 1)
    lo, hi, u = (a, b, t * 2) if t < 0.5 else (b, c, (t - 0.5) * 2)
    return "#" + "".join(f"{round(x + (y - x) * u):02x}" for x, y in zip(lo, hi))


def hatch(pattern_id, color):
    return (f'<defs><pattern id="{pattern_id}" width="6" height="6" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="2" height="6" fill="{color}"/></pattern></defs>')


def slot_ranks_svg(runs=RUNS):
    cases = {}
    for run in runs:
        for row in load_rows(run):
            cases.setdefault(row["suite/case"], row["suite/task"])
    def size(cid):
        p = dict(get_case(cases[cid], cid).params)
        return (p["tokens"] * p["hidden"] * p["rank"], p["hidden"], cid)
    ids = sorted(cases, key=size)
    width, left, row_h, top = 860, 300, 30, 72
    height = top + len(ids) * row_h + 60
    sx = lambda r: left + ((r.bit_length() - 1) - 2) / 7 * (width - left - 20)  # noqa: E731 (log2, 4..512)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" {FONT} font-size="11">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', hatch("pad", MUTED),
           f'<text x="12" y="20" font-size="14" font-weight="600" fill="{INK}">Per-slot ranks for every case in the evidence</text>',
           f'<text x="12" y="36" fill="{INK2}">Bar = rank per slot (log₂ axis); hatched = padding to a multiple of 16 (NaN-poisoned).</text>',
           f'<text x="12" y="51" fill="{INK2}">No hatched part means no padding, so the wrong_padding control has nothing to catch (not applicable).</text>']
    for r in (8, 16, 32, 64, 128, 256):
        x = sx(r)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{top + len(ids) * row_h}" stroke="{GRID}"/>'
                   f'<text x="{x:.1f}" y="{top - 8}" text-anchor="middle" fill="{MUTED}">{r}</text>')
    for i, cid in enumerate(ids):
        task = cases[cid]
        p = dict(get_case(task, cid).params)
        ranks = slot_ranks(p)
        y = top + i * row_h
        out.append(f'<text x="12" y="{y + 18}" fill="{INK2}">{escape(label(task, cid))}</text>')
        bar_h = max(1.5, (row_h - 8) / len(ranks))
        for j, r in enumerate(ranks):
            padded = (r + 15) // 16 * 16
            yy = y + 4 + j * bar_h
            h = max(1.2, bar_h - 0.8)
            out.append(f'<rect x="{left}" y="{yy:.2f}" width="{sx(r) - left:.2f}" height="{h:.2f}" fill="{ramp(j, len(ranks))}">'
                       f'<title>slot {j}: rank {r}{f", padded to {padded}" if padded > r else ", no padding"}</title></rect>')
            if padded > r:
                out.append(f'<rect x="{sx(r):.2f}" y="{yy:.2f}" width="{sx(padded) - sx(r):.2f}" height="{h:.2f}" fill="url(#pad)"/>')
    ly = top + len(ids) * row_h + 26
    out.append(f'<rect x="12" y="{ly - 9}" width="10" height="10" fill="{SEQ[0]}"/><text x="26" y="{ly}" fill="{INK2}">slot 0</text>'
               f'<rect x="80" y="{ly - 9}" width="10" height="10" fill="{SEQ[1]}"/><text x="94" y="{ly}" fill="{INK2}">middle slots</text>'
               f'<rect x="180" y="{ly - 9}" width="10" height="10" fill="{SEQ[2]}"/><text x="194" y="{ly}" fill="{INK2}">last slot</text>'
               f'<rect x="270" y="{ly - 9}" width="10" height="10" fill="url(#pad)"/><text x="284" y="{ly}" fill="{INK2}">rank padding</text>')
    out.append(f'<text x="12" y="{ly + 22}" fill="{MUTED}">Cases from the {len(runs)} completed Stage 6 runs (smoke, smoke rerun, acceptance, stratified); '
               'ranks via utils.lora_contracts.slot_ranks.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n", len(ids)


def cell(row):
    """(fill, text, text colour) for one suite row."""
    status, reason = row["suite/status"], str(row.get("suite/reason") or "")
    if status == "passed" and row["suite/fixture"] == "honest":
        tier = TIER.get(row.get("reward/reason"), 1)
        return SEQ[tier - 1], str(tier), INK if tier == 1 else "#ffffff"
    if status == "passed":
        return GOOD, "✓", "#ffffff"
    if status == "failed":
        return CRITICAL, "✗", "#ffffff"
    if reason.startswith("control_not_applicable_"):
        return "url(#na)", "n/a", MUTED
    return WARN, "?", INK


def coverage_svg(run=COVERAGE_RUN):
    rows = load_rows(run)
    summary = json.loads((ROOT / "artifacts" / "runs" / run / "summary.json").read_text())
    counts = summary["counts"]
    not_applicable = sum(r["suite/status"] == "unavailable"
                         and str(r.get("suite/reason") or "").startswith("control_not_applicable_") for r in rows)
    by_key = {(r["suite/task"], r["suite/case"], r["suite/backend"], r["suite/fixture"]): r for r in rows}
    groups = []
    for r in rows:
        key = (r["suite/task"], r["suite/case"])
        if key not in groups:
            groups.append(key)
    def size(key):
        p = dict(get_case(*key).params)
        return (LORA_TASKS.index(key[0]), p["tokens"] * p["hidden"] * p["rank"], key[1])
    groups.sort(key=size)
    cols = [(b, f) for b in BACKENDS for f in FIXTURES]
    cw, ch, left, top = 34, 24, 420, 228
    width = left + len(cols) * cw + 20
    height = top + len(groups) * ch + 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" {FONT} font-size="11">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', hatch("na", GRID),
           f'<text x="12" y="20" font-size="14" font-weight="600" fill="{INK}">Coverage matrix · Stratified</text>',
           f'<text x="12" y="37" fill="{INK2}">{escape(run)} · {counts["passed"]} passed, {counts["failed"]} failed, '
           f'{counts["unavailable"]} unavailable ({not_applicable} planned not-applicable)</text>']
    legend = [(SEQ[2], "honest · tier 3 faster_than_both"), (SEQ[1], "tier 2 faster_than_eager"),
              (SEQ[0], "tier 1 correct"), (GOOD, "control caught ✓"), (CRITICAL, "failed ✗"),
              ("url(#na)", "not applicable"), (WARN, "unavailable (other)")]
    for n, (color, text) in enumerate(legend):  # two rows of four/three, fixed slots
        x, y = 12 + (n % 4) * 190, 52 + (n // 4) * 18
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" fill="{color}" stroke="{GRID}"/>'
                   f'<text x="{x + 14}" y="{y}" fill="{INK2}">{escape(text)}</text>')
    for k, (b, f) in enumerate(cols):
        cx = left + k * cw + cw / 2
        out.append(f'<text transform="translate({cx + 4:.1f},{top - 6}) rotate(-90)" fill="{INK2}">{b} · {f}</text>')
    out.append(f'<text x="12" y="{top - 6}" fill="{MUTED}">task</text><text x="140" y="{top - 6}" fill="{MUTED}">case</text>')
    for i, (task, cid) in enumerate(groups):
        y = top + i * ch
        out.append(f'<text x="12" y="{y + 16}" fill="{INK2}">{task.replace("lora_", "")}</text>'
                   f'<text x="140" y="{y + 16}" fill="{INK2}">{escape(label(task, cid))}</text>')
        for k, (b, f) in enumerate(cols):
            row = by_key.get((task, cid, b, f))
            if row is None:
                continue
            fill, text, color = cell(row)
            cx = left + k * cw
            tip = f'{task} · {b} · {f} · {row["suite/status"]}' + (f' · {row["reward/reason"]}' if row.get("reward/reason") else "")
            out.append(f'<rect x="{cx + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="3" fill="{fill}">'
                       f'<title>{escape(tip)}</title></rect>'
                       f'<text x="{cx + cw / 2:.1f}" y="{y + 16}" text-anchor="middle" font-size="10.5" font-weight="600" fill="{color}">{text}</text>')
    out.append(f'<text x="12" y="{top + len(groups) * ch + 24}" fill="{MUTED}">Honest cells show the reward tier earned; '
               'control cells show whether the control was caught. Source: suite/progress.jsonl.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n", len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "stage6")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ranks, n_cases = slot_ranks_svg()
    (args.out / "fig_stage6_slot_ranks.svg").write_text(ranks, encoding="utf-8")
    coverage, n_rows = coverage_svg()
    (args.out / "fig_stage6_coverage_stratified.svg").write_text(coverage, encoding="utf-8")
    print(json.dumps({"slot_ranks_cases": n_cases, "coverage_rows": n_rows, "out": str(args.out)}))


if __name__ == "__main__":
    main()
