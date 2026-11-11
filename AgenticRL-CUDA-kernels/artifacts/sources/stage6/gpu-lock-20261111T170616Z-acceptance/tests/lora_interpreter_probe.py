"""Synthetic small probes. CPU execution is correctness evidence only."""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.support import ROOT


def load_kernel():
    spec=importlib.util.spec_from_file_location('stage6_lora_kernel',ROOT/'tasks/lora_shrink_bgmv/kernels/triton/lora.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    from tests.test_lora_stage6 import tiny
    from utils.lora_contracts import LORA_TASKS
    from utils.lora_reference import make_inputs, forward
    from utils.lora_verification import check_outputs
    candidate=load_kernel()
    torch.set_num_threads(2)
    for dtype in ('float16','bfloat16'):
        for pattern in ('mixed','none','single','distinct','round_robin'):
            for name in LORA_TASKS:
                args=make_inputs(name,tiny(dtype,pattern))
                actual=candidate.forward(name,*args)
                expected=forward(name,*args)
                check_outputs(name,actual,expected,args)
                print(name,dtype,pattern,'PASS',flush=True)
    print('STAGE6_INTERPRETER_OK')


if __name__=='__main__':
    main()
