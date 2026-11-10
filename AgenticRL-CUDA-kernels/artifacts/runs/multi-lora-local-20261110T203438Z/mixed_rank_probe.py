import json
from pathlib import Path
import torch
from tests.lora_interpreter_probe import load_kernel
from utils.task_catalog import case
from utils.lora_contracts import LORA_TASKS
from utils.lora_reference import make_inputs,forward
from utils.lora_verification import check_outputs
candidate=load_kernel();torch.set_num_threads(2);torch.manual_seed(29)
rows=[]
for dtype in ('float16','bfloat16'):
    for name in LORA_TASKS:
        item=case(dtype,tokens=17,hidden=32,out=19,rank=256,slots=16,slices=3,pattern='mixed',mixed=True)
        args=make_inputs(name,item)
        report=check_outputs(name,candidate.forward(name,*args),forward(name,*args),args)
        rows.append({'case':item.to_dict(),'report':report,'device':'cpu-interpreter'})
        print(name,dtype,'passed',flush=True)
Path('/tmp/stage6-repair/mixed-rank-probes.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
