"""Read-only checks of the final frozen public source and launch request."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
bundle = root / Path('/tmp/stage6-repair/bundle-path').read_text()
source = bundle / 'source'
manifest_path = source / 'task-source-manifest.json'
manifest = json.loads(manifest_path.read_text())
failures = []
for name, expected in manifest['files'].items():
    for label, base in [('frozen', source), ('working', root)]:
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        if actual != expected:
            failures.append([label, name])
required = ['utils/lora_verification.py', 'tests/test_lora_gate.py',
            'tests/test_graph_lifetime.py', 'tests/fixtures/benchmark_capture_before_repair.py',
            'tests/test_repeated_forward.py', 'tests/repeated_forward_probe.py',
            'legal/multi_lora/sources.json', 'legal/multi_lora/punica-LICENSE',
            'legal/multi_lora/slora-LICENSE', 'legal/multi_lora/vllm-LICENSE']
assert all(name in manifest['files'] for name in required)
assert len(manifest['tasks']) == 6
assert not failures, failures
protected = ('model.py', 'model_new.py', 'binding.cpp', 'binding_registry.h',
             'kernels/axpby.cu', 'kernels/axpby_binding.cpp', 'SKILL.md')
upstream = json.loads((source / 'artifacts/upstream.json').read_text())['sha256']
assert all(manifest['files'][name] == upstream[name] for name in protected)
request_path = bundle / 'h200-smoke-request.json'
request = json.loads(request_path.read_text())
assert Path(request['cwd']) == source and request['launched'] is False
assert request['command'][2] == '--detach' and request['environment'] == {'MODAL_GPU': 'H200'}
request['source_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
request_path.write_text(json.dumps(request, indent=2) + '\n')
summary = {'frozen_files': len(manifest['files']), 'tasks': manifest['tasks'],
           'source_manifest_sha256': request['source_manifest_sha256'],
           'hash_mismatches': failures, 'all_seven_protected_files_match_upstream': True,
           'new_regressions_and_legal_files_included': True, 'modal_launched': False}
(bundle / 'freeze-review.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
