"""Torch-free Stage 6 run sizing. Estimates are planning assumptions, not timings."""

import argparse
import json
from pathlib import Path

from utils.lora_contracts import LORA_TASKS, slot_ranks, token_slots
from utils.task_catalog import select_cases


FIXTURES = ('honest', 'hack', 'wrong', 'wrong_padding', 'wrong_skip')


def size_case(name, item):
    """Conservative explicit-tensor budget, including graph pools and float64 gate.

    This is not an allocator peak guarantee: compiler/runtime workspaces are
    unknown until a GPU run. All byte counts use full padded physical shapes.
    """
    p = item.params
    t, h, n, r, l, s = (p[k] for k in ('tokens', 'hidden', 'out', 'rank', 'slots', 'slices'))
    r = (r + 15) // 16 * 16
    parts = dict(x=2*t*h, a=2*l*s*r*h, b=2*l*s*n*r,
                 y=2*t*s*(n+16), buf=4*s*t*r,
                 idx=4*t, ranks=4*l, scales=4*l, offsets=4*s,
                 metadata=4*(t+3*l+1))
    if name == 'lora_segment_metadata':
        inputs = parts['idx'] + parts['ranks']
        output = parts['metadata']
        builder = inputs
        workspace = 64*t*l + 64*t
        gate = 0
    else:
        fields = (('x', 'a', 'b', 'y', 'idx', 'ranks', 'scales', 'offsets') if 'fused' in name
                  else ('x', 'a', 'idx', 'ranks', 'scales') if 'shrink' in name
                  else ('buf', 'b', 'y', 'idx', 'ranks', 'offsets'))
        inputs = sum(parts[key] for key in fields)
        if name.endswith('sgmv'):
            inputs += parts['metadata']
        # The builder constructs both weights and all x/y/buf arrays, including
        # arrays not returned. Account for simultaneous random/division buffers.
        builder = sum(parts.values()) + 2*max(parts['a'], parts['b'])
        output = parts['buf'] if 'shrink' in name else parts['y']
        # Full per-slot reference matmuls, fp32 weights, x, masks, slice gathers,
        # copies and fp32 scratch. Each term is a padded, physical shape.
        workspace = (4*s*t*r + 4*t*h + 8*s*r*h if 'shrink' in name else
                     4*s*t*n + 8*s*n*r + 32*t*n + 8*s*t*r)
        if 'fused' in name:
            workspace += 4*s*t*r + 4*t*h + 8*s*r*h
        # Full gate, not a sample: two double selections, error, normalization,
        # masks, sorting values/indices and temporary reductions. Includes y gaps.
        elements = s*t*r if 'shrink' in name else t*s*(n+16)
        gate = 96*elements
    reserve = 16*1024**3
    budget = 8*builder + 12*workspace + 12*output + 2*gate + reserve
    return dict(input_bytes=inputs, output_bytes=output, builder_peak_budget_bytes=builder,
                reference_workspace_budget_bytes=workspace, gate_workspace_budget_bytes=gate,
                gpu_planning_budget_bytes=budget, gpu_capacity_bytes=141_000_000_000,
                gpu_headroom_bytes=141_000_000_000-budget,
                scope='Explicit-tensor planning budget, not measured allocator/compiler peak',
                assumptions='8 builder/input copies; 12 workspace/output pools; 2 full gate workspaces; 16 GiB runtime reserve',
                skipped_tokens=token_slots(p).count(-1),
                rank_padding=min(slot_ranks(p)) < r)


def planned_rows(selection):
    rows = []
    for name in LORA_TASKS:
        for item in select_cases(name, selection):
            size = size_case(name, item)
            for backend in ('cuda', 'triton'):
                for fixture in FIXTURES[:3] if name == 'lora_segment_metadata' else FIXTURES:
                    applicable = (fixture != 'wrong_skip' or size['skipped_tokens'] > 0)
                    applicable &= fixture != 'wrong_padding' or size['rank_padding']
                    rows.append(dict(task=name, case=item.id, backend=backend, fixture=fixture,
                                     applicable=applicable))
    return rows


def plan(smoke):
    """Project from the actual smoke rows with explicit extra large-case budgets."""
    smoke_rows = json.loads((Path(smoke)/'suite/suite.json').read_text())['rows']
    wall = {(r['suite/task'], r['suite/backend'], r['suite/fixture']): r['suite/wall_s']
            for r in smoke_rows if 'suite/wall_s' in r}
    result = {'schema': 'stage6-gpu-plan-v1', 'smoke_evidence': str(smoke),
              'gpu_timings_measured_for_new_cases': False, 'runs': {}}
    for selection in ('acceptance', 'stratified'):
        cases = {name: [dict(case=c.to_dict(), sizing=size_case(name, c))
                        for c in select_cases(name, selection)] for name in LORA_TASKS}
        rows = planned_rows(selection)
        baseline = sum(wall[(r['task'], r['backend'],
                             'wrong' if r['fixture'] == 'wrong_skip' else r['fixture'])]
                       for r in rows if r['applicable'])
        # At the widest shape, reserve additional 300 s per honest/hack row and
        # 120 s per verification-only control, linearly with output/weight volume.
        # This pays for unmeasured compiler specialization, CPU random builders,
        # repeated full gates/sorts and uncached graph pools. It is an assumption.
        extra = 0.
        for row in rows:
            if not row['applicable'] or row['task'] == 'lora_segment_metadata':
                continue
            entry = next(c for c in cases[row['task']] if c['case']['id'] == row['case'])
            p = entry['case']['parameters']
            scale = (p['hidden']/14336)*(p['rank']/256)*(p['slices']/3)*max(p['tokens']/4096, .05)
            extra += scale * (300 if row['fixture'] in ('honest', 'hack') else 120)
        estimate = 1.5*baseline + extra + 300
        result['runs'][selection] = dict(
            cases=cases, rows=rows, total_rows=len(rows), applicable_rows=sum(r['applicable'] for r in rows),
            smoke_baseline_projection_s=baseline, extra_large_case_budget_s=extra,
            estimated_wall_s=estimate, per_case_timeout_s=1200,
            suite_budget_s=20700, modal_function_limit_s=21600,
            estimate_within_suite_budget=estimate < 20700,
            time_model='1.5 * matched smoke row wall + explicit shape-scaled extra budget + 300s startup',
            time_limitations='Estimate only. 600s is not demonstrated for new shapes; request 1200s per case. Cold compile, profiling/sort cost and allocator peaks remain GPU checks.')
    result['decode_finding'] = (
        'The paired catalog already includes 256-token BGMV at hidden=4096. '
        'The last/widest acceptance row is 64 tokens at hidden=14336. '
        'No catalog change; combined 256-token/widest shape is not claimed.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(plan(args.smoke), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
