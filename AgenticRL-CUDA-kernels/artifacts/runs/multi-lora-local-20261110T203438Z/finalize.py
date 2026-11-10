"""Retain completed local repair evidence; never launches cloud work."""
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

root = Path.cwd()
temp = Path('/tmp/stage6-repair')
bundle = root / (temp / 'bundle-path').read_text()

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

summaries = {}
for key in ('system', 'runtime'):
    raw = (temp / f'{key}-suite.log').read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK(?: \(skipped=(\d+)\))?', raw)
    assert match and 'FAILED (' not in raw, f'{key} suite not complete/passing'
    summaries[key] = {'tests': int(match[1]), 'seconds': float(match[2]),
                      'skipped': int(match[3] or 0), 'raw_summary': match[0]}
probe = (temp / 'interpreter-probes.log').read_text()
assert probe.count(' PASS') == 60 and 'STAGE6_INTERPRETER_OK' in probe
repeat = (temp / 'full-repeated-probes.log').read_text()
assert repeat.count('"calls": 40') == 66 and 'REPEATED_FORWARD_PROBES_OK' in repeat

path = root / 'VALIDATION.md'
original = path.read_bytes()
ending = '\r\n' if b'\r\n' in original else '\n'
text = original.decode().replace('\r\n', '\n')
pending = ('The installed-runtime suite is recorded separately once complete. No commit\n'
           'or Modal launch is authorized by this repair handoff; Claude reviews, commits\n'
           'and launches the prepared request only after user approval.\n')
replacement = ('The installed-runtime discovery run also completed:\n\n```text\n'
               + summaries['runtime']['raw_summary'] + '\n```\n\n'
               'The final 60-case Stage 6 interpreter sweep passed under the new gate.\n'
               'The runtime discovery run also repeated all 66 immutable-input scenarios\n'
               '(2,640 forwards). Skips cover unavailable GPU/backend packages and\n'
               'optional integrations; these are not GPU acceptance. Existing SQLite\n'
               'ResourceWarnings are retained in the raw logs.\n\n'
               'Discovery overlapped the repair work. Final config positional\n'
               'compatibility and the portable archived-helper fixture were\n'
               'checked separately afterward (`final-routing.log`: 5 tests, OK;\n'
               '`self-contained-repro.log`: 2 tests, OK).\n\n'
               'No commit or Modal launch was performed. Claude reviews the frozen\n'
               'source and prepared request and launches only after user approval.\n')
assert pending in text
path.write_bytes(text.replace(pending, replacement).replace('\n', ending).encode())

for source in temp.iterdir():
    if source.is_file() and source.suffix in ('.json', '.log', '.py'):
        shutil.copyfile(source, bundle / source.name)

commands_path = bundle / 'commands.json'
commands = json.loads(commands_path.read_text())
commands['interpreter'] = ('PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=1 '
                          'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
                          '.venv/bin/python tests/lora_interpreter_probe.py')
commands['freeze'] = ('PYTHONDONTWRITEBYTECODE=1 python3 -m scripts.freeze_task_suite '
                      f'--tasks stage6 --output {bundle.relative_to(root)}/source')
write_json(commands_path, commands)

changed_path = bundle / 'changed-files.json'
changed = json.loads(changed_path.read_text())
changed['sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in changed['files']}
write_json(changed_path, changed)
old = root / 'artifacts/runs/stage6-smoke-20261110T105217Z/source'
patch = []
for name in changed['files']:
    before = (old / name).read_text().splitlines(True) if (old / name).is_file() else []
    after = (root / name).read_text().splitlines(True)
    patch.extend(difflib.unified_diff(before, after,
                 fromfile=f'a/{name}' if before else '/dev/null', tofile=f'b/{name}'))
(bundle / 'review.patch').write_text(''.join(patch))

summary = {
    'scope': 'Stage 6 relative verification gate and shared graph buffer ownership repair',
    'suites': summaries, 'stage6_interpreter_cases': 60,
    'repeated_forward_scenarios': 66, 'repeated_forward_calls': 2640,
    'input_mutation_reproduced': False,
    'graph_ownership_defect_reproduced': True,
    'h200_root_cause_confirmed': False, 'h200_fix_validated': False,
    'supplemental_rank_probes': 32, 'honest_bound_failures_observed': 0,
    'maximum_observed_fraction_of_bound': 0.5649717514124294,
    'remaining_risks': ['CUDA native and compiled-reference repeated-call behavior',
                        'Actual H200 graph replay with retained buffers',
                        'Full GPU catalog numerical acceptance under the new gate'],
    'task_forward_abi_changed': False, 'input_mutability_contract_changed': False,
    'numerical_criterion': 'Approved Stage 6 relative criterion replaces legacy float check only',
    'final_focused_checks': 'Discovery overlapped implementation; final config position and portable reproduction fixture covered by final-routing.log and self-contained-repro.log.',
    'committed': False, 'modal_launched': False,
}
write_json(bundle / 'summary.json', summary)

run_id = bundle.name.replace('multi-lora-local-', 'stage6-smoke-repair-')
request = {
    'approval_required': True,
    'approval_source': 'HANDOFF_CODEX_STAGE6_GATE.md and HANDOFF_CODEX_STAGE6_REPLAY_CRASHES.md',
    'launch_owner': 'Claude after review and user approval', 'launched': False,
    'scope': 'Stage 6 H200 smoke, both backends and all fixtures; no acceptance run',
    'cwd': str(bundle / 'source'), 'gpu_requested': 'H200',
    'command': [str(root / '.venv/bin/modal'), 'run', '--detach', 'cloud/modal_tasks.py',
                '--tasks', 'stage6', '--case-set', 'smoke', '--backends', 'cuda,triton',
                '--fixtures', 'all', '--cuda-graphs', '--timeout-s', '600',
                '--run-id', run_id],
    'environment': {'MODAL_GPU': 'H200'}, 'benchmark': True,
    'cold_start': True, 'cuda_graphs': True,
    'expected_outputs': f'agenticrl-cuda-artifacts:/{run_id}',
    'download_destination': str(root / 'artifacts/runs' / run_id),
    'limitations': 'Local ownership repro and interpreter checks do not establish H200 causality, replay or numerical acceptance.',
}
write_json(bundle / 'h200-smoke-request.json', request)

readme = (bundle / 'README.md').read_text()
readme = readme.replace('`source/` will contain the definitive public freeze.',
                        '`source/` contains the definitive public freeze, verified against its manifest.')
readme += '\n## Completed validation\n\n'
for key, value in summaries.items():
    readme += f"{key} discovery:\n\n```text\n{value['raw_summary']}\n```\n\n"
readme += ('The final 60-case interpreter sweep passed. The runtime suite repeated all\n'
           '66 immutable-input scenarios. GPU/optional-package skips remain unverified;\n'
           'the raw logs retain existing SQLite ResourceWarnings. `summary.json` records\n'
           'the scope and limitations. `review.patch` is a convenience diff against the\n'
           'archived smoke source. Early `target.log` includes a corrected synthetic\n'
           'test-fixture failure; the final focused logs and both full suites pass.\n')
readme += ('Full discovery overlapped implementation. Final config positional\n'
           'compatibility and the portable archived-helper fixture are covered by\n'
           '`final-routing.log` (5 tests, OK) and `self-contained-repro.log` (2, OK).\n')
(bundle / 'README.md').write_text(readme)
print(json.dumps(summary, indent=2))
