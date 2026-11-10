"""Record strict integer/float CPU interpreter checks; never report GPU acceptance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.task_catalog import get_task
from utils.task_suite import ROOT, scaffold, select_tasks
from utils.rollout_env import bootstrap_script


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in ('.py','.cu','.cpp','.h')
            and not any(part in ('.cache','__pycache__','build') for part in p.relative_to(path).parts)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default='stage5')
    parser.add_argument('--fixtures', default='honest,chain_replay')
    parser.add_argument('--case-set', default='smoke')
    parser.add_argument('--timeout-s',type=float,default=600)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.timeout_s <= 0:parser.error('timeout must be positive')
    names=select_tasks(args.tasks.split(','))
    if not names or any(get_task(n).stage != 5 for n in names):parser.error('Stage 5 tasks required')
    variants=args.fixtures.split(',')
    if not variants or set(variants)-{'honest','chain_replay','hack','wrong','wrong_ancestor'}:
        parser.error('unknown fixture')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'schema':'stage5-cpu-verification-v2','created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'CPU interpreter correctness only; no CUDA build, GPU timing or provenance acceptance',
              'argv':sys.argv,'harness_sha256':hashes(ROOT/'utils'),'rows':[]}
    from utils.task_catalog import select_cases
    for name in names:
        for item in select_cases(name,args.case_set):
            for variant in variants:
                if variant=='chain_replay' and name!='eagle_gdn_tree_recurrence':continue
                if variant=='wrong_ancestor' and name=='eagle_candidate_topk':continue
                path=scaffold(name,'triton',variant,args.output/name/item.id/variant)
                (path/'.cache').mkdir(exist_ok=True)
                command=[sys.executable,'-B','-u','-c',bootstrap_script('utils.task_verification',[],path.resolve())]
                env={**os.environ,'CUDA_AGENT_TASK':name,'CUDA_AGENT_TASK_CASE':item.id,
                     'CUDA_AGENT_DEVICE':'cpu','TRITON_INTERPRET':'1','PYTHONDONTWRITEBYTECODE':'1',
                     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                record={'task':name,'case':item.id,'variant':variant,'argv':command,
                        'environment':{k:env[k] for k in ('CUDA_AGENT_TASK','CUDA_AGENT_TASK_CASE','CUDA_AGENT_DEVICE','TRITON_INTERPRET','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                        'source_sha256':hashes(path),'log':str(path/'verification.log')}
                started=time.monotonic()
                with (path/'verification.log').open('w') as stream:
                    proc=subprocess.Popen(command,cwd=path,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code=proc.wait(timeout=args.timeout_s)
                        record.update(status='verified' if code==0 else 'failed',exit_code=code)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        record.update(status='timeout',exit_code=None)
                record['wall_s']=time.monotonic()-started
                manifest['rows'].append(record)
                (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
                print(json.dumps(record,allow_nan=False),flush=True)
    return int(any(row['status']!='verified' for row in manifest['rows']))


if __name__=='__main__':
    raise SystemExit(main())
