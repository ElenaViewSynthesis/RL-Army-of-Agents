import json
import torch
from tests.lora_interpreter_probe import load_kernel
from utils.task_catalog import case
from utils.lora_contracts import LORA_TASKS
from utils.lora_reference import make_inputs, forward
from utils.lora_diagnostics import error_distribution
candidate=load_kernel()
torch.set_num_threads(2)
torch.manual_seed(17)
rows=[]
for rank in (64,256):
    for dtype in ('float16','bfloat16'):
        for name in LORA_TASKS:
            if name=='lora_segment_metadata':
                continue
            item=case(dtype,tokens=7,hidden=32,out=19,rank=rank,slots=4,slices=3,pattern='mixed',mixed=False)
            args=make_inputs(name,item)
            actual=candidate.forward(name,*args)
            expected=forward(name,*args)
            torch.testing.assert_close(actual,expected,atol=1e-2,rtol=1e-2)
            half=forward(name,*args,half=True)
            row={'task':name,'rank':rank,'dtype':dtype,'device':'cpu','interpreter':True,
                 'shape_scope':'synthetic T7/H32/N19/L4/S3; not catalog GPU evidence',
                 'honest':error_distribution(actual,expected,getattr(torch,dtype)),
                 'fp16_intermediate':error_distribution(half,expected,getattr(torch,dtype)),
                 'fp16_intermediate_expected_failure':False,'status':'passed'}
            rows.append(row)
            print(json.dumps(row),flush=True)
with open('/tmp/lora-stage6/rank-probe.json','w') as stream:
    json.dump({'rows':rows,'diagnostic_only':True},stream,indent=2,allow_nan=False)
print('STAGE6_RANK_PROBES_OK',flush=True)
