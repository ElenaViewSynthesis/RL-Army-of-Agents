"""Complete local evidence after both suites finish; does not launch services."""
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

root = Path.cwd()
temp = Path('/tmp/stage6-gpu-lock')
bundle = root / (temp/'bundle-path').read_text()
def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

summaries = {}
for name in ('system', 'runtime'):
    raw = (temp/f'{name}-suite.log').read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK(?: \(skipped=(\d+)\))?', raw)
    assert match and 'FAILED (' not in raw, f'{name} suite incomplete or failed'
    summaries[name] = dict(tests=int(match[1]), seconds=float(match[2]),
                           skipped=int(match[3] or 0), raw_summary=match[0])
sizes = json.loads((temp/'symbolic-sizing.json').read_text())
assert len(sizes['rows']) == 39
plan = json.loads((temp/'run-plan.json').read_text())
assert all(r['estimate_within_suite_budget'] for r in plan['runs'].values())

p = root/'VALIDATION.md'
b = p.read_bytes(); newline = '\r\n' if b'\r\n' in b else '\n'
text = b.decode().replace('\r\n', '\n')
pending = ('sources have not run on a GPU. Required suite summaries will be appended when\n'
           'both discovery processes complete.')
assert pending in text
text = text.replace(pending, 'sources have not run on a GPU. Both required discovery runs completed:')
text += '\n\n'
for name, value in summaries.items():
    text += f"{name} interpreter:\n\n```text\n{value['raw_summary']}\n```\n\n"
text += ('Full logs are retained. Skips reflect absent Torch/backends or GPUs and\n'
         'opt-in live integrations; they are not passing hardware checks. Existing\n'
         'SQLite ResourceWarnings are retained. The final driver-node fallback\n'
         'branch was also checked by `final-lock.log` (9 tests, OK), after full\n'
         'discovery had begun. No prior/reaped test job was restarted.\n\n'
         'The final lock/run-plan checks also pass with the installed runtime\n'
         'under python -S: 12 tests in 2.699 s, OK (`final-runtime-parent.log`).\n\n'
         'All 39 full-size symbolic scenarios completed; maximum visible builder\n'
         'storage was 1199968332 bytes and reference-plus-fixture live storage was\n'
         '3593734672 bytes. These exclude hidden operator/allocator workspaces;\n'
         'the larger planning budget remains the relevant GPU capacity estimate.\n\n'
         'Each prepared request points to its own fresh public source freeze and\n'
         'manifest SHA-256. Claude launches acceptance, reports, then launches\n'
         'stratified only after review/approval. No Modal/W&B write or commit was made.\n')
p.write_bytes(text.replace('\n', newline).encode())

for source in temp.iterdir():
    if source.is_file() and source.suffix in ('.json', '.log', '.py'):
        shutil.copyfile(source, bundle/source.name)

names = list(json.loads((temp/'before-hashes.json').read_text()))
names += ['utils/gpu_lock.py', 'scripts/plan_stage6.py', 'scripts/size_stage6.py',
          'tests/test_gpu_lock.py', 'tests/test_stage6_plan.py']
changed = [name for name in names if not (temp/'before'/name).is_file()
           or (root/name).read_bytes() != (temp/'before'/name).read_bytes()]
patch = []
for name in changed:
    before = (temp/'before'/name).read_text().splitlines(True) if (temp/'before'/name).is_file() else []
    patch += difflib.unified_diff(before, (root/name).read_text().splitlines(True),
                  fromfile='a/'+name if before else '/dev/null', tofile='b/'+name)
(bundle/'review.patch').write_text(''.join(patch))
dump(bundle/'changed-files.json', {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                   for name in changed})
upstream = json.loads((root/'artifacts/upstream.json').read_text())['sha256']
protected = ('SKILL.md', 'model.py', 'model_new.py', 'binding.cpp', 'binding_registry.h',
             'kernels/axpby.cu', 'kernels/axpby_binding.cpp')
checks = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() == upstream[name]
          for name in protected}
assert all(checks.values())
summary = dict(scope='Stage 6 GPU preparation and per-device measurement lock',
               suites=summaries, symbolic_scenarios=len(sizes['rows']),
               maximum_symbolic_live_bytes=3593734672,
               maximum_memory_planning_budget_bytes=106799303296,
               protected_upstream_checks=checks, new_dependencies=False,
               compute_reward_inputs_changed=False, only_new_reward_reason='gpu_busy',
               only_new_advantage_exclusion='gpu_busy', new_default_result_keys=6,
               lock_default_enabled=True, lock_wait_counts_toward_task_deadline=False,
               task_forward_abi_changed=False, verification_gate_changed=False,
               gpu_validation_of_new_sources=False, modal_launched=False,
               live_tracking_write=False, committed=False,
               remaining_checks=['Real GPU contention/visibility lookup in deployment',
                                 'Actual allocator/compiler peaks and cold-run deadline margins',
                                 'bf16, mixed-rank, skipped-row and largest-case GPU correctness/graphs'])
dump(bundle/'summary.json', summary)
stamp = bundle.name.removeprefix('stage6-gpu-lock-local-')
sources = {selection: root/'artifacts/sources/stage6'/f'gpu-lock-{stamp}-{selection}'
           for selection in ('acceptance', 'stratified')}
dump(bundle/'freeze-targets.json', {k: str(v) for k, v in sources.items()})
commands = json.loads((bundle/'commands.json').read_text())
for selection, source in sources.items():
    commands[f'freeze_{selection}'] = (f'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m '
                                       f'scripts.freeze_task_suite --tasks stage6 --output {source}')
dump(bundle/'commands.json', commands)
readme = ('# Stage 6 GPU preparation and GPU contention lock — local evidence\n\n'
          'No Modal run, live tracking write or commit was performed. Claude owns\n'
          'review/commit and launches each detached GPU run after user approval.\n\n'
          'The lock serializes profiling, benchmark and diagnostics by device UUID,\n'
          'with an explicit visibility-key fallback. Shared lock directory defaults\n'
          'to the system temporary directory/agenticrl-gpu-locks and is overridden\n'
          'with CUDA_AGENT_GPU_LOCK_DIR. CPU/disabled paths do not acquire. Only\n'
          'lock wait extends the task deadline; capped wait is gpu_busy, excluded\n'
          'within original advantage groups. Other reasons/reward inputs are unchanged.\n'
          'Auxiliary ceiling timing also locks; numerical LoRA checks are verification.\n\n'
          '[RUN_PLAN.md](RUN_PLAN.md) lists every case ID, control expectation and\n'
          'resource assumption. [run-plan.json](run-plan.json) enumerates every row.\n'
          'The stratified covering subset has six cases per task, 336 rows (312\n'
          'applicable); acceptance has 56/56 applicable. Estimated wall is 3.91 h\n'
          'and 2.38 h respectively, with 1200 s per-case caps. Neither is launched.\n'
          'The widest BGMV acceptance uses T64; the catalog already has T256 at\n'
          'hidden4096. No catalog shape was added or removed.\n\n'
          'The CPU symbolic probe checks 39 largest-case reference/control storage\n'
          'scenarios. Authored kernels use a reference-storage proxy, not device\n'
          'execution. Float64 gate/quantile arrays and graph/runtime reserves are\n'
          'budgeted separately: maximum 106.799 GB, headroom 34.201 GB of 141 GB.\n'
          'GPU allocator/compiler peaks, timeout margins, new-case correctness\n'
          'and real contention remain unverified. Full source hashes are recorded\n'
          'per request after each fresh freeze.\n\n'
          '## Validation\n\n')
for name, value in summaries.items():
    readme += f"{name}:\n\n```text\n{value['raw_summary']}\n```\n\n"
readme += ('The final 9 lock/exclusion tests pass (3.339 s); the wider focused run\n'
           'passes 71 tests (44.261 s, 6 skips). Exact commands and full logs are\n'
           'retained. GPU/optional runtime skips are not hardware passes. Protocol\n'
           'timings and device names in unit-test logs are synthetic. The driver-node\n'
           'fallback branch was checked separately after full discovery started.\n'
           'The installed-runtime parent checks under python -S also pass: 12\n'
           'tests in 2.699 s, OK (`final-runtime-parent.log`).\n'
           'Initial-target logs retain intermediate missing-tracking-row failures,\n'
           'and plan-target retains the corrected GB/GiB test-threshold mismatch;\n'
           'the memory model was not reduced to pass that check. Final runs pass.\n\n'
           '[changed-files.json](changed-files.json) and [review.patch](review.patch)\n'
           'describe the source edits. All seven protected files match upstream.\n')
(bundle/'README.md').write_text(readme)
print(json.dumps(dict(bundle=str(bundle), summaries=summaries, freeze_targets={k:str(v) for k,v in sources.items()}), indent=2))
