"""Review the downloaded immutable smoke evidence; never modify a result or reward."""
import collections,hashlib,json
from pathlib import Path
project=Path(__file__).resolve().parents[3]
root=project/'artifacts/runs/stage6-smoke-20261110T105217Z'
suite=json.loads((root/'suite/suite.json').read_text())
summary=json.loads((root/'summary.json').read_text())
manifest=json.loads((root/'source/task-source-manifest.json').read_text())
missing_or_changed=[p for p,h in manifest['files'].items()
                    if not (root/'source'/p).is_file() or hashlib.sha256((root/'source'/p).read_bytes()).hexdigest()!=h]
honest=[r for r in suite['rows'] if r['suite/fixture']=='honest']
bench_bad=[{'task':r['suite/task'],'backend':r['suite/backend'],'reason':r.get('bench/unavailable_reason')}
           for r in honest if r.get('bench/available') is not True]
graphs=[{'task':r['suite/task'],'backend':r['suite/backend'],
         **{k:r.get('bench/'+k) for k in ('eager/graph_capturable','eager/graph_replay_ok',
             'compile/graph_capturable','compile/graph_replay_ok','generated/graph_capturable','generated/graph_replay_ok')}}
        for r in honest]
tasks={n:dict(collections.Counter(r['suite/status'] for r in suite['rows'] if r['suite/task']==n))
       for n in dict.fromkeys(r['suite/task'] for r in suite['rows'])}
review={'run_id':summary['run_id'],'suite_counts':suite['counts'],'per_task':tasks,
        'source_mismatches_remote':summary['source_mismatches'],'source_mismatches_downloaded':missing_or_changed,
        'honest_rows':len(honest),'honest_gates_passed':sum(r['suite/status']=='passed' for r in honest),
        'honest_provenance':[{'task':r['suite/task'],'backend':r['suite/backend'],'fraction':r.get('task/provenance_authored')} for r in honest],
        'benchmark_unavailable':bench_bad,'graphs':graphs,
        'unavailable_reasons':dict(collections.Counter(r['suite/reason'] for r in suite['rows'] if r['suite/status']=='unavailable')),
        'scope':'Frozen FP16 smoke cases only; not acceptance, bf16 or full-grid coverage.',
        'complete_validation_ok':not missing_or_changed and not summary['source_mismatches'] and not suite['counts']['failed'] and not bench_bad
               and all(all(r[k] is True for k in ('eager/graph_capturable','eager/graph_replay_ok','compile/graph_capturable','compile/graph_replay_ok','generated/graph_capturable','generated/graph_replay_ok')) for r in graphs)}
(root/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
print(json.dumps(review,indent=2))
