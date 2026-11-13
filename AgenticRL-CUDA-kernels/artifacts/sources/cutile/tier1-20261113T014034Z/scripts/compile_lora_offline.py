"""Compile authored Stage 6 Triton kernels without a GPU; never execution evidence."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('lora_offline',root/'tasks/lora_shrink_bgmv/kernels/triton/lora.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=[]
    def compile_one(arch,name,pointers,constants):
        fn=getattr(module,name)
        signature=dict(zip(fn.arg_names,pointers))
        triton.compile(ASTSource(fn,signature=signature,constexprs=constants),target=GPUTarget('cuda',arch,32))
        row={'architecture':f'sm_{arch}','kernel':name,'pointers':pointers,'constants':constants,'status':'compiled','device_executed':False}
        rows.append(row)
        print(json.dumps(row),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'triton':triton.__version__,'rows':rows,'limitation':'Compilation only; no device execution'},indent=2)+'\n')
    for arch in (75,90):
        compile_one(arch,'lora_count_slots',['*i32','*i32'],dict(T=4096,L=16,BLOCK=4096))
        compile_one(arch,'lora_group_tokens',['*i32']*7,dict(T=4096,L=16,BLOCK=4096,LS=32))
        for dtype in ('fp16','bf16'):
            p='*'+dtype
            compile_one(arch,'lora_initialize',[p,p],dict(N=4096,COPY=True,BLOCK=256))
            compile_one(arch,'lora_decode_shrink',[p,p,'*i32','*i32','*fp32','*fp32'],dict(T=3,H=2048,R=16,S=3,SPLITS=2,K=1024))
            compile_one(arch,'lora_merge_shrink',['*fp32']*2,dict(N=144,SPLITS=2,BS=2,BLOCK=64))
            compile_one(arch,'lora_decode_expand',['*fp32',p,p,'*i32','*i32','*i32',p],dict(T=3,N=2048,R=256,S=3,W=6192,RN=256,BN=16))
            for shrink in (True,False):
                pointers=[p if shrink else '*fp32',p,p,'*i32','*fp32' if shrink else '*i32','*i32',*(['*i32']*5),'*fp32' if shrink else p]
                compile_one(arch,'lora_segment_gemm',pointers,dict(T=4096,H=14336 if shrink else 0,R=256,S=3,N=14336,W=43056,L=16,SHRINK=shrink,M=16,BN=16,BK=32))


if __name__=='__main__':
    main()
