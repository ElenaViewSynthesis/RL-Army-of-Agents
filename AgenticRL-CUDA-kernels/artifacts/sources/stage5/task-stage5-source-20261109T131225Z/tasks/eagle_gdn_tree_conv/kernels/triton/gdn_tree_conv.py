# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch
import triton
import triton.language as tl


@triton.jit
def eagle_ancestor_table(Parents, Paths, N: tl.constexpr, BLOCK: tl.constexpr):
    batch, node = tl.program_id(0), tl.program_id(1)
    current = node
    for depth in range(N):
        tl.store(Paths + (batch*N+node)*N+depth, current)
        current = tl.load(Parents+batch*N+tl.maximum(current,0), current>=0, other=-1)


def paths_for(parents):
    b,n=parents.shape
    paths=torch.empty((b,n,n),device=parents.device,dtype=torch.int32)
    eagle_ancestor_table[(b,n)](parents,paths,n,triton.next_power_of_2(n))
    return paths

@triton.jit
def eagle_conv_windows(Parents, Windows, N:tl.constexpr):
    batch,node=tl.program_id(0),tl.program_id(1)
    current=node
    for depth in range(4):
        tl.store(Windows+(batch*N+node)*4+depth,current)
        parent=tl.load(Parents+batch*N+tl.maximum(current,0),current>=0,other=-1)
        current=tl.where(current>=0,parent,current-1)


@triton.jit
def eagle_causal_conv(X,History,W,Bias,Windows,Out,N:tl.constexpr,C:tl.constexpr,MAMBA:tl.constexpr,BLOCK:tl.constexpr):
    batch,node=tl.program_id(0),tl.program_id(1)
    channels=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    total=tl.full((BLOCK,),0,tl.float32)
    for offset in range(4):
        source=tl.load(Windows+(batch*N+node)*4+offset)
        address=batch*N*C+tl.where(MAMBA,channels*N+tl.maximum(source,0),tl.maximum(source,0)*C+channels)
        value=tl.load(X+address,(channels<C)&(source>=0),other=0).to(tl.float32)
        history=tl.load(History+(batch*3+source+3)*C+channels,(channels<C)&(source<0),other=0).to(tl.float32)
        weight=tl.load(W+channels*4+3-offset,channels<C,other=0)
        total=total+(value+history)*weight
    bias=tl.load(Bias+channels,channels<C,other=0)
    address=batch*N*C+tl.where(MAMBA,channels*N+node,node*C+channels)
    tl.store(Out+address,total+bias,channels<C)


def forward(x,history,weight,bias,parents,mamba=False):
    b,n=parents.shape;c=weight.shape[0]
    windows=torch.empty((b,n,4),device=x.device,dtype=torch.int32)
    output=torch.empty_like(x,dtype=torch.float32)
    eagle_conv_windows[(b,n)](parents,windows,n)
    eagle_causal_conv[(b,n,triton.cdiv(c,128))](x,history,weight,bias,windows,output,n,c,mamba,128,enable_fp_fusion=False)
    return output,parents,windows
