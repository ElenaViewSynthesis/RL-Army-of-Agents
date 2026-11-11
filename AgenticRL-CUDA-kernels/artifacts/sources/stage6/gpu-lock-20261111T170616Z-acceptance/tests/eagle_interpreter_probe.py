"""Synthetic branching probes, separate from catalog smoke acceptance."""
import os,sys,importlib.util,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from utils import eagle_reference as ref
from utils.task_catalog import case
from utils.task_verification import check_equal

def kernel(name):
 p=ROOT/'tasks'/name/'kernels/triton'/f'{name.removeprefix("eagle_")}.py'
 spec=importlib.util.spec_from_file_location('_probe_'+name,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
for shape in ('chain','wide','balanced','ragged'):
 for n in (1,8):
  p=dict(b=2,n=n,tree=shape,hkv=1,ratio=4,d=64,s=512,window=512,c=192,heads=1,dk=64,dv=64,v=33,seed=17,top_k=4)
  for name,fn in [('eagle_tree_step',ref.tree_step),('eagle_tree_row_attention',ref.attention),('eagle_gdn_tree_conv',ref.tree_conv),('eagle_mamba2_tree_conv',ref.tree_conv),('eagle_gdn_tree_recurrence',ref.recurrence),('eagle_accept_tail',ref.accept),('eagle_accept_tail_sampled',ref.accept_sampled)]:
   dtype='bfloat16' if 'conv' in name or 'attention' in name else 'int32' if name=='eagle_accept_tail' else 'float32'
   args=ref.make_inputs(name,case(dtype,**p));kwargs={'window':512} if 'attention' in name else {'mamba':True} if 'mamba2' in name else {}
   torch.manual_seed(42)
   expected=fn(*args,**kwargs)
   m=kernel(name)
   actual=m.forward(*args,**kwargs)
   check_equal(actual,expected)
   if name=='eagle_gdn_tree_recurrence':
    chain=m.forward(*args,chain=True);check_equal(chain,expected);check_equal(chain,actual)
   print(json.dumps({'task':name,'tree':shape,'n':n,'status':'pass'}),flush=True)
# Small deterministic selection probes exercise the same GPU kernel path.
m=kernel('eagle_candidate_topk')
for v in (17,1031):
 x=torch.ones((2,64),dtype=torch.bfloat16);w=torch.ones((v,64),dtype=torch.bfloat16)
 check_equal(m.forward(x,w,8),ref.candidate_topk(x,w,8))
 print(json.dumps({'task':'eagle_candidate_topk','v':v,'status':'ties-pass'}),flush=True)

for v in (17,1031):
 torch.manual_seed(123)
 x=torch.randn((2,64),dtype=torch.bfloat16);w=torch.randn((v,64),dtype=torch.bfloat16)
 check_equal(m.forward(x,w,8),ref.candidate_topk(x,w,8))
 print(json.dumps({'task':'eagle_candidate_topk','v':v,'status':'random-pass'}),flush=True)
print("STAGE5_INTERPRETER_PROBES_OK",flush=True)
