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

__global__ void eagle_cuda_frontier(const float* scores,const float* children,const int* parents,float* out,int* ids,int* paths,int* flat,int n,int k){
    int b=blockIdx.x;int used[64];
    for(int rank=0;rank<n;rank++){
        float best=-INFINITY;int selected=INT_MAX;
        for(int i=0;i<n*k;i++){
            bool seen=false;for(int j=0;j<rank;j++)seen|=used[j]==i;
            float value=__fadd_rn(scores[b*n+i/k],children[b*n*k+i]);
            if(!seen&&(value>best||(value==best&&i<selected))){best=value;selected=i;}
        }used[rank]=selected;out[b*n+rank]=best;ids[b*n+rank]=selected/k;flat[b*n+rank]=selected;
        int current=selected/k;paths[(b*n+rank)*(n+1)]=n+rank;
        for(int depth=0;depth<n;depth++){paths[(b*n+rank)*(n+1)+depth+1]=current;current=current>=0?parents[b*n+current]:-1;}
    }
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),n=x[0].size(1),k=x[1].size(2);
    auto out=torch::empty_like(x[0]);auto ids=torch::empty_like(x[2]);auto flat=torch::empty_like(x[2]);auto paths=torch::empty({b,n,n+1},x[2].options());
    eagle_cuda_frontier<<<b,1,0,stream>>>(x[0].data_ptr<float>(),x[1].data_ptr<float>(),x[2].data_ptr<int>(),out.data_ptr<float>(),ids.data_ptr<int>(),paths.data_ptr<int>(),flat.data_ptr<int>(),n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,ids,paths,flat};
}
