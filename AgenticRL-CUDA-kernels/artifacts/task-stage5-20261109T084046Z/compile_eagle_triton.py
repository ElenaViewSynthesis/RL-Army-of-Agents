"""Offline compiler probes: emits H200 code, never initializes a GPU driver or launches a kernel."""
from pathlib import Path
import os,sys,importlib.util,json,hashlib,traceback
os.environ['TRITON_CACHE_DIR']='/tmp/eagle-stage5-triton-cache'
os.environ['TRITON_INTERPRET']='0'
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
ROOT=Path('/mnt/c/Users/proxi/Documents/codex-9/CUDA-RL/AgenticRL-CUDA-kernels')
sys.path.insert(0,str(ROOT))
from utils.task_catalog import TASKS
out=ROOT/'artifacts/task-stage5-20261109T084046Z/offline-triton-compile.json'
records=[];seen=set();current={}

def capture(fn,grid):
 def launch(*args,**kwargs):
  signature={};constants={}
  for param,arg in zip(fn.params,args):
   if param.is_constexpr:constants[param.name]=arg
   elif isinstance(arg,torch.Tensor):signature[param.name]={torch.float32:'*fp32',torch.bfloat16:'*bf16',torch.int32:'*i32',torch.int64:'*i64',torch.float64:'*fp64'}[arg.dtype]
   else:signature[param.name]='i32'
  key=(fn.cache_key,str(signature),str(constants))
  if key in seen:return
  seen.add(key)
  record={**current,'kernel':fn.__name__,'signature':signature,'constants':constants,'target':'cuda sm90 warp32','scope':'offline compiler only; no execution'}
  try:
   compiled=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',90,32),options={'num_warps':4,'enable_fp_fusion':False})
   record.update(status='compiled',shared_bytes=compiled.metadata.shared,ptx_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest())
  except Exception as exc:
   record.update(status='failed',error=traceback.format_exc())
  records.append(record);out.write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps({k:v for k,v in record.items() if k not in ('signature','constants','error')}),flush=True)
  if record['status']=='failed':print(record['error'],flush=True)
 return launch
triton.runtime.jit.JITFunction.__getitem__=capture
T=lambda *s,dtype=torch.float32:torch.empty(s,device='meta',dtype=dtype)
for name,task in TASKS.items():
 if task.stage!=5:continue
 path=ROOT/'tasks'/name/'kernels/triton'/f'{name.removeprefix("eagle_")}.py'
 spec=importlib.util.spec_from_file_location('_offline_'+name,path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 for large in (False,True):
  b,n,d=(32,64,128) if large else (1,1,64)
  current={'task':name,'case':'large' if large else 'smoke','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  if name=='eagle_candidate_topk':m.forward(T(b,4096,dtype=torch.bfloat16),T(151936 if large else 32000,4096,dtype=torch.bfloat16),16 if large else 8)
  elif name=='eagle_tree_step':m.forward(T(b,n),T(b,n,8 if large else 4),T(b,n,dtype=torch.int32))
  elif 'attention' in name:
   hq,hk,s=(64,8,32768) if large else (8,8,512)
   m.forward(T(b,hq,n,d,dtype=torch.bfloat16),T(b,hk,s,d,dtype=torch.bfloat16),T(b,hk,s,d,dtype=torch.bfloat16),T(b,hk,n,d,dtype=torch.bfloat16),T(b,hk,n,d,dtype=torch.bfloat16),T(b,n,dtype=torch.int32),min(s,4096) if name=='eagle_tree_row_attention' else s)
  elif 'conv' in name:
   c=384 if large else 192;mamba='mamba2' in name
   m.forward(T(b,c,n,dtype=torch.bfloat16) if mamba else T(b,n,c,dtype=torch.bfloat16),T(b,3,c,dtype=torch.bfloat16),T(c,4),T(c),T(b,n,dtype=torch.int32),mamba=mamba)
  elif 'recurrence' in name:
   h=4 if large else 1;args=(T(b,n,h,d),T(b,n,h,d),T(b,n,h,d),T(b,n,h,2),T(b,h,d,d),T(b,n,dtype=torch.int32))
   m.forward(*args);m.forward(*args,chain=True)
  elif name=='eagle_accept_tail':m.forward(T(b,n,dtype=torch.int32),T(b,n,dtype=torch.int32),T(b,n+1,dtype=torch.int32),T(b,n,dtype=torch.int32))
  else:
   v=151936 if large else 32000
   m.forward(T(b,n,dtype=torch.int32),T(b,n,dtype=torch.int32),T(b,n+1,v),T(b,n+1,v),T(b,n,dtype=torch.int32),T(b,n),T(b,n+1))
print('COMPILATION_DONE',len(records),sum(r['status']=='failed' for r in records),flush=True)
