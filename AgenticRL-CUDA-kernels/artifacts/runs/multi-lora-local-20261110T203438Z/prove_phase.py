import json
from pathlib import Path
root=Path('artifacts/runs/stage6-smoke-20261110T105217Z')
source=(root/'source/utils/benchmark.py').read_text().splitlines()
rows=json.loads((root/'suite/suite.json').read_text())['rows']
proof=[]
for row in rows:
    if row['suite/fixture']=='honest' and not row.get('bench/available'):
        stderr=row['log/stderr_tail']
        assert 'line 160, in run_benchmark' in stderr
        assert '_time_calls(graph.replay, args.iters_per_round)' in stderr
        assert '_time_calls(graph.replay, args.iters_per_round)' in source[159]
        proof.append({'task':row['suite/task'],'backend':row['suite/backend'],
                      'phase':'timed graph replay call, followed by synchronize',
                      'frozen_source_line':160,'implementation_that_faulted':'not identified by traceback'})
print(json.dumps(proof,indent=2))
