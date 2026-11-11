import torch,json,time
from utils.lora_reference import make_inputs,forward
from utils.lora_contracts import LORA_TASKS
from utils.task_catalog import select_cases
for name in LORA_TASKS:
 item=select_cases(name,'acceptance')[0];start=time.monotonic()
 with torch.device('meta'):
  args=make_inputs(name,item);out=forward(name,*args)
 print(name,[list(t.shape) for t in args],list(out.shape) if isinstance(out,torch.Tensor) else [list(t.shape) for t in out],time.monotonic()-start,flush=True)
