from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,subprocess,time,concurrent.futures,hashlib
ROOT=Path('/mnt/c/Users/proxi/Documents/codex-9/CUDA-RL/AgenticRL-CUDA-kernels')
sys.path.insert(0,str(ROOT))
from utils.task_catalog import TASKS
from utils.task_suite import scaffold
run=ROOT/'artifacts'/('task-stage5-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
run.mkdir();(ROOT/'artifacts/task-stage5-current.txt').write_text(str(run)+'\n')
jobs=[(n,'honest') for n,t in TASKS.items() if t.stage==5]+[('eagle_gdn_tree_recurrence','chain_replay')]
def verify(job):
 name,variant=job;path=scaffold(name,'triton',variant,run/name/variant)
 env={**os.environ,'CUDA_AGENT_TASK':name,'CUDA_AGENT_TASK_CASE':TASKS[name].cases[0].id,'CUDA_AGENT_DEVICE':'cpu','TRITON_INTERPRET':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 script=f"import sys,runpy;sys.path.insert(0,{str(ROOT)!r});import utils;sys.path.insert(0,{str(path)!r});runpy.run_module('utils.task_verification',run_name='__main__')"
 command=[str(ROOT/'.venv/bin/python'),'-B','-c',script]
 start=time.monotonic()
 with (path/'verification.log').open('w') as stream:
  try:
   p=subprocess.run(command,cwd=path,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
   status='verified' if p.returncode==0 else 'failed'
   code=p.returncode
  except subprocess.TimeoutExpired:status='timeout';code=None
 record={'task':name,'variant':variant,'case':TASKS[name].cases[0].id,'status':status,'exit_code':code,'wall_s':time.monotonic()-start,'argv':command,'environment':{k:env[k] for k in ('CUDA_AGENT_TASK','CUDA_AGENT_TASK_CASE','CUDA_AGENT_DEVICE','TRITON_INTERPRET','OMP_NUM_THREADS','MKL_NUM_THREADS')},'log':str(path/'verification.log')}
 print(json.dumps(record),flush=True)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 records=list(pool.map(verify,jobs))
manifest={'schema':'stage5-cpu-verification-v1','rows':records,'scope':'CPU interpreter correctness only; no GPU timing, provenance or CUDA build acceptance','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for name,t in TASKS.items() if t.stage==5 for p in (ROOT/'tasks'/name).rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cpp','.h')}}
(run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
