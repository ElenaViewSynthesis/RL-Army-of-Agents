import json,os,time
from pathlib import Path
from tests.support import scratch
from utils.task_catalog import get_task
from utils.task_suite import scaffold,case_environment
from utils.rollout_env import _stage
rows=[]
with scratch('relative-verifier-') as tmp:
    for variant in ('honest','wrong_padding','wrong_skip'):
        name='lora_shrink_bgmv';item=get_task(name).cases[2]
        path=scaffold(name,'triton',variant,Path(tmp)/variant)
        (path/'.cache').mkdir()
        with case_environment(name,item.id,'cpu'):
            code,stdout,stderr,timed_out=_stage('utils.lora_verification',['--seed','17'],path,os.environ.copy(),time.monotonic()+600)
        report=json.loads((path/'.cache/relative-error-bounds.json').read_text())
        assert not timed_out
        assert report['passed'] == (variant=='honest'),report
        assert (code==0) == (variant=='honest'),(code,stderr)
        rows.append({'variant':variant,'exit_code':code,'report':report,'stdout':stdout,'stderr':stderr})
        print(variant,'exit',code,'passed',report['passed'],flush=True)
Path('/tmp/stage6-repair/verifier-integration.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
