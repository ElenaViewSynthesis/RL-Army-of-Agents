"""Repeated synthetic CPU forwards: immutable inputs, stable outputs, no GPU evidence."""
import importlib.util
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.task_catalog import case
from utils.lora_contracts import LORA_TASKS
from utils.lora_reference import make_inputs,forward
from utils.lora_verification import bit_equal,check_outputs


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def clone(value):
    if isinstance(value,torch.Tensor):return value.clone()
    return type(value)(clone(x) for x in value)


def equal(a,b):
    if isinstance(a,torch.Tensor):return bit_equal(a,b)
    return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))


def repeat(label,fn,args,expected,gate=None):
    before=clone(args);baseline=None;last=None
    for i in range(40):
        actual=fn(*args)
        assert equal(args,before),(label,'mutated input',i)
        if gate:gate(actual,expected,args)
        else:
            from utils.task_verification import check_equal
            check_equal(actual,expected)
        if baseline is None:baseline=clone(actual)
        assert equal(actual,baseline),(label,'unstable output',i)
        if last is not None:assert equal(last,baseline),(label,'previous output overwritten',i)
        last=actual
    print(json.dumps({'label':label,'calls':40,'input_bytes_unchanged':True,'outputs_bit_stable':True}),flush=True)


def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    candidate=load(ROOT/'tasks/lora_shrink_bgmv/kernels/triton/lora.py','repeat_lora_kernel')
    for dtype in ('float16','bfloat16'):
        for pattern in ('mixed','none'):
            for name in LORA_TASKS:
                item=case(dtype,tokens=7,hidden=16,out=9,rank=16,slots=4,slices=2,pattern=pattern,mixed=True)
                args=make_inputs(name,item)
                module=load(ROOT/'tasks'/name/'model.py','repeat_reference_'+name)
                reference=module.Model();expected=reference(*args)
                gate=lambda a,b,x,n=name:check_outputs(n,a,b,x)
                repeat(f'{name}/{dtype}/{pattern}/reference',reference,args,expected,gate)
                repeat(f'{name}/{dtype}/{pattern}/triton',lambda *x,n=name:candidate.forward(n,*x),args,expected,gate)
    from utils import eagle_reference as ref
    functions={'eagle_tree_step':ref.tree_step,'eagle_candidate_topk':ref.candidate_topk,
               'eagle_tree_row_attention':ref.attention,'eagle_tree_verify_attention':ref.attention,
               'eagle_gdn_tree_conv':ref.tree_conv,'eagle_mamba2_tree_conv':ref.tree_conv,
               'eagle_gdn_tree_recurrence':ref.recurrence,'eagle_accept_tail':ref.accept,
               'eagle_accept_tail_sampled':ref.accept_sampled}
    p=dict(b=1,n=3,tree='balanced',h=64,v=17,top_k=2,hkv=1,ratio=1,d=64,s=16,window=16,c=8,heads=1,dk=64,dv=64,seed=17)
    for name,fn in functions.items():
        dtype='bfloat16' if 'attention' in name or 'conv' in name or 'topk' in name else 'int32' if name=='eagle_accept_tail' else 'float32'
        args=ref.make_inputs(name,case(dtype,**p))
        kwargs={'window':16} if 'attention' in name else {'mamba':True} if 'mamba2' in name else {}
        if name=='eagle_candidate_topk':kwargs={'top_k':2}
        expected=fn(*args,**kwargs)
        module=load(ROOT/'tasks'/name/'kernels/triton'/f'{name.removeprefix("eagle_")}.py','repeat_'+name)
        repeat(name+'/reference',lambda *x,f=fn,k=kwargs:f(*x,**k),args,expected)
        repeat(name+'/triton',lambda *x,m=module,k=kwargs:m.forward(*x,**k),args,expected)
    print('REPEATED_FORWARD_PROBES_OK',flush=True)


if __name__=='__main__':main()
