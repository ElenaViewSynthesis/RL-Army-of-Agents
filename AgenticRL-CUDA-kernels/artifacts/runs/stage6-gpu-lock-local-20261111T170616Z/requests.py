"""Verify independent freezes and write exact requests; never invokes Modal."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
bundle = root/Path('/tmp/stage6-gpu-lock/bundle-path').read_text()
def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
targets = json.loads((bundle/'freeze-targets.json').read_text())
plan = json.loads((bundle/'run-plan.json').read_text())
smoke = json.loads((root/plan['smoke_evidence']/'suite/suite.json').read_text())['rows']
smoke_wall = {(r['suite/task'], r['suite/backend'], r['suite/fixture']): r['suite/wall_s']
              for r in smoke if 'suite/wall_s' in r}
stamp = bundle.name.removeprefix('stage6-gpu-lock-local-')
reviews = {}
for index, (selection, location) in enumerate(targets.items(), 1):
    source = Path(location)
    manifest_path = source/'task-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    mismatches = [f'{label}:{name}' for label, base in [('frozen', source), ('working', root)]
                  for name, digest in manifest['files'].items()
                  if hashlib.sha256((base/name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    for name in ('utils/gpu_lock.py', 'scripts/plan_stage6.py', 'scripts/size_stage6.py',
                 'tests/test_gpu_lock.py', 'tests/test_stage6_plan.py',
                 'legal/multi_lora/sources.json', 'legal/multi_lora/punica-LICENSE',
                 'legal/multi_lora/slora-LICENSE', 'legal/multi_lora/vllm-LICENSE'):
        assert name in manifest['files'], name
    run = plan['runs'][selection]
    row_estimates = []
    for row in run['rows']:
        if not row['applicable']:
            continue
        entry = next(c for c in run['cases'][row['task']] if c['case']['id'] == row['case'])
        p = entry['case']['parameters']
        extra = 0.0
        if row['task'] != 'lora_segment_metadata':
            scale = (p['hidden']/14336)*(p['rank']/256)*(p['slices']/3)*max(p['tokens']/4096, .05)
            extra = scale*(300 if row['fixture'] in ('honest', 'hack') else 120)
        fixture = 'wrong' if row['fixture'] == 'wrong_skip' else row['fixture']
        row_estimates.append(1.5*smoke_wall[(row['task'], row['backend'], fixture)] + extra)
    assert max(row_estimates) < 1200
    run_id = f'stage6-{selection}-{stamp}'
    request = dict(approval_required=True, approval_source='HANDOFF_CODEX_STAGE6_GPU_AND_LOCK.md',
                   launch_owner='Claude after review/user approval', launch_order=index,
                   stop_after_run_and_report=True, automatic_rerun=False, launched=False,
                   cwd=str(source), environment={'MODAL_GPU':'H200', 'PYTHONPATH':str(source)},
                   gpu_requested='H200',
                   command=[str(root/'.venv/bin/modal'), 'run', '--detach', 'cloud/modal_tasks.py',
                            '--tasks', 'stage6', '--case-set', selection, '--backends', 'cuda,triton',
                            '--fixtures', 'all', '--cuda-graphs', '--timeout-s', '1200', '--run-id', run_id],
                   run_id=run_id, benchmark=True, cold_start=True, cuda_graphs=True,
                   source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   source_files=len(manifest['files']), source_git_revision=manifest['git_revision'],
                   case_ids={name:[c['case']['id'] for c in cases] for name,cases in run['cases'].items()},
                   estimated_wall_s=run['estimated_wall_s'], estimate_model=run['time_model'],
                   estimate_limitations=run['time_limitations'],
                   timeout_s=1200, gpu_lock=True, gpu_lock_timeout_s=600,
                   maximum_projected_row_s=max(row_estimates),
                   projected_row_budget_below_case_cap=True,
                   expected_outcome_counts={'passed':run['applicable_rows'], 'failed':0,
                                            'unavailable':run['total_rows']-run['applicable_rows']},
                   expected_outcomes={
                       'honest':'verify under relative gate (metadata exact), authored >=0.9, zero library time',
                       'hack':'verification succeeds; provenance fails',
                       'wrong':'verification fails',
                       'wrong_padding':'verification fails where padded ranks exist; unavailable only without padding',
                       'wrong_skip':'verification fails with idx=-1; unavailable only without skipped tokens',
                       'fp16_intermediate':'diagnostic attached to honest rows, never an expected-failure row',
                       'gpu_busy':'environment unavailable, excluded from training advantages; report any occurrence'},
                   additional_honest_checks='bench available; eager/compile/generated graph capture and replay successful; report failures separately from reward outcomes',
                   maximum_gpu_planning_budget_bytes=max(c['sizing']['gpu_planning_budget_bytes']
                                                         for cases in run['cases'].values() for c in cases),
                   gpu_capacity_bytes=141000000000,
                   memory_limitations='Symbolic/static estimates with graph, full verifier and runtime reserves; actual compiler/allocator peak not measured',
                   expected_outputs=f'agenticrl-cuda-artifacts:/{run_id}',
                   download_destination=str(root/'artifacts/runs'/run_id),
                   tracking_owner='Claude downloads JSON/log evidence and logs once to W&B after review',
                   evidence_plan=str(bundle/'RUN_PLAN.md'))
    dump(bundle/f'{selection}-request.json', request)
    reviews[selection] = dict(source=str(source), source_files=len(manifest['files']),
                              manifest_sha256=request['source_manifest_sha256'], mismatches=mismatches,
                              new_code_tests_legal_files_present=True, launched=False)
assert targets['acceptance'] != targets['stratified']
dump(bundle/'freeze-review.json', reviews)
with (bundle/'README.md').open('a') as f:
    f.write('\n## Prepared requests and source freezes\n\n')
    for selection, review in reviews.items():
        f.write(f"- [{selection}-request.json]({selection}-request.json): {review['source_files']} files, manifest SHA-256 `{review['manifest_sha256']}`. Zero frozen/working hash mismatches.\n")
    f.write('\nBoth requests remain `launched: false`. Each has a distinct fresh source\nfreeze, exact detached argv, case IDs, wall estimate and expected outcomes.\n')
summary = json.loads((bundle/'summary.json').read_text())
summary['freezes'] = reviews
dump(bundle/'summary.json', summary)
for name, digest in json.loads((bundle/'changed-files.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
print(json.dumps(reviews, indent=2))
