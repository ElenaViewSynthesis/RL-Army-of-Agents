"""Sequential, fresh-process/cache captures. Diagnostic timings never feed reward."""
import argparse
import json
import os
import time
from pathlib import Path

from tests.support import make_task, scratch
from utils.backend import BACKENDS
from utils.rollout_env import _stage


def plan(backends, repeat=None):
    return [(backend, i, 'gen,eager,compile' if i % 2 == 0 else 'eager,compile,gen')
            for backend in backends for i in range(repeat if repeat is not None else (100 if backend == 'cutedsl' else 20))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends', default=','.join(BACKENDS))
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    backends = args.backends.split(',')
    if any(b not in BACKENDS for b in backends) or (args.repeat is not None and args.repeat <= 0):
        parser.error('known backends and positive repeat required')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_AGENT_DEVICE='cuda',TRITON_INTERPRET='0')
    rows = []
    for backend, index, order in plan(backends,args.repeat):
        out = args.output_dir / f'{backend}-{index:03d}'
        out.mkdir()
        row = {'backend':backend,'repeat':index,'order':order,'stages':{}}
        with scratch('timing-') as tmp:
            task = make_task(tmp,backend)
            (task/'.cache').mkdir(exist_ok=True)
            import hashlib
            row['candidate_sha256'] = hashlib.sha256((task/'model_new.py').read_bytes()).hexdigest()
            (out/'model_new.py').write_bytes((task/'model_new.py').read_bytes())
            env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            # Override inherited cache settings: every repetition is cold and independent.
            for spec in BACKENDS.values():
                for var, subdir in spec.cache_env:
                    env[var] = str(task/'.cache'/subdir)
            env['TORCHINDUCTOR_CACHE_DIR'] = str(task/'.cache/inductor')
            row['cache_env'] = {k:v for k,v in env.items() if k.endswith('CACHE_DIR')}
            # Only the extension needs a separate build. JIT first launch belongs in
            # this fresh profiling process's warmup, just as in the failing session.
            stages = ([('compile','utils.compile',['--backend',backend])] if backend == 'cuda' else [])
            stages += [('profiling','utils.profiling',['--backend',backend,'--json','--order',order,
                                                     '--trace-dir',str(out)])]
            for stage, module, argv in stages:
                start = time.monotonic()
                code, stdout, stderr, timeout = _stage(module,argv,task,env,start+600)
                elapsed = time.monotonic()-start
                (out/f'{stage}.stdout.log').write_text(stdout)
                (out/f'{stage}.stderr.log').write_text(stderr)
                row['stages'][stage] = {'returncode':code,'timed_out':timeout,'wall_s':elapsed}
                if code != 0 or timeout:
                    row['reason'] = f'{stage}_failed'
                    break
                if stage == 'profiling':
                    try:
                        row['profile'] = json.loads(stdout.strip().splitlines()[-1])
                    except (ValueError,IndexError):
                        row['reason'] = 'malformed_profile'
        rows.append(row)
        (out/'record.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        # Append progress for remote monitoring and preserve partial work on timeout.
        with (args.output_dir/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'backend':backend,'repeat':index,'timing_measured':row.get('profile',{}).get('timing_measured'),
                          'reason':row.get('reason') or row.get('profile',{}).get('timing_reason') }),flush=True)
    failures = [r for r in rows if not r.get('profile',{}).get('timing_measured')]
    summary = {'command':'python -m scripts.diagnose_timing','backends':backends,'repetitions':len(rows),
               'empty_captures':[r for r in rows if str(r.get('profile',{}).get('timing_reason')).startswith('empty_profile:')],
               'failures':failures,'rows':rows,'diagnostic_only':True}
    (args.output_dir/'timing-diagnostic.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
