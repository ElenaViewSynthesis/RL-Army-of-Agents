// SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
// Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
// See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.

#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <climits>
using Tensor = torch::Tensor;
using bf16 = __nv_bfloat16;
__device__ float eagle_reduce(float x) {
    __shared__ float sums[32];
    for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
    if(!(threadIdx.x&31)) sums[threadIdx.x/32]=x;
    __syncthreads();
    x=threadIdx.x<blockDim.x/32?sums[threadIdx.x]:0.f;
    if(threadIdx.x<32) for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
    if(threadIdx.x==0) sums[0]=x;
    __syncthreads();float y=sums[0];__syncthreads();return y;
}
__global__ void eagle_cuda_paths(const int* parents,int* paths,int n) {
    int b=blockIdx.x,node=blockIdx.y,current=node;
    for(int i=0;i<n;i++){paths[(b*n+node)*n+i]=current;current=current>=0?parents[b*n+current]:-1;}
}
static void eagle_check(const std::vector<Tensor>& x) {
    TORCH_CHECK(!x.empty(),"inputs required");
    for(const auto& t:x) TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==x[0].device(),"same-device contiguous CUDA tensors required");
}

__global__ void eagle_cuda_accept(const int* tokens,const int* parents,const int* targets,const int* slots,int* lengths,int* paths,int* bonus,int* compact,int n){
    int b=blockIdx.x,current=-1,length=0;for(int i=0;i<n;i++){paths[b*n+i]=-1;compact[b*n+i]=-1;}
    for(int depth=0;depth<n;depth++){
        int wanted=targets[b*(n+1)+current+1],chosen=-1;
        for(int j=0;j<n;j++)if(parents[b*n+j]==current&&tokens[b*n+j]==wanted){chosen=j;break;}
        if(chosen<0)break;paths[b*n+length]=chosen;compact[b*n+length]=slots[b*n+chosen];length++;current=chosen;
    }lengths[b]=length;bonus[b]=targets[b*(n+1)+current+1];
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),n=x[0].size(1);auto lengths=torch::empty({b},x[0].options());auto bonus=torch::empty_like(lengths);auto paths=torch::empty_like(x[0]);auto compact=torch::empty_like(x[3]);
    eagle_cuda_accept<<<b,1,0,stream>>>(x[0].data_ptr<int>(),x[1].data_ptr<int>(),x[2].data_ptr<int>(),x[3].data_ptr<int>(),lengths.data_ptr<int>(),paths.data_ptr<int>(),bonus.data_ptr<int>(),compact.data_ptr<int>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {lengths,paths,bonus,compact,x[1]};
}
