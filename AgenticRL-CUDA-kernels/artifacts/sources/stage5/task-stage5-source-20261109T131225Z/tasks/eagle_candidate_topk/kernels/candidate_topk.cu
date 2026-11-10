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

__global__ void eagle_cuda_vocab(const bf16* x,const bf16* w,float* scores,int h,int v){
    int b=blockIdx.x,token=blockIdx.y;float total=0;
    for(int i=threadIdx.x;i<h;i+=blockDim.x) total=__fadd_rn(total,__fmul_rn(__bfloat162float(x[b*h+i]),__bfloat162float(w[token*h+i])));
    float sum=eagle_reduce(total);if(threadIdx.x==0)scores[b*v+token]=sum;
}
__global__ void eagle_cuda_topk(const float* scores,int64_t* ids,float* rescored,const bf16* x,const bf16* w,int h,int v,int k){
    int b=blockIdx.x;
    // A conservative serial selector; no performance acceptance is claimed.
    if(threadIdx.x==0)for(int rank=0;rank<k;rank++){
        float best=-INFINITY;int chosen=INT_MAX;
        for(int token=0;token<v;token++){
            bool used=false;for(int j=0;j<rank;j++)used|=ids[b*k+j]==token;
            float score=scores[b*v+token];
            if(!used&&(score>best||(score==best&&token<chosen))){best=score;chosen=token;}
        }ids[b*k+rank]=chosen;
    }
    __syncthreads();
    for(int rank=0;rank<k;rank++){
        int token=ids[b*k+rank];float sum=0;
        for(int i=threadIdx.x;i<h;i+=blockDim.x)sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(x[b*h+i]),__bfloat162float(w[token*h+i])));
        sum=eagle_reduce(sum);if(threadIdx.x==0)rescored[b*k+rank]=sum;
    }
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),h=x[0].size(1),v=x[1].size(0),k=parameter;
    auto scores=torch::empty({b,v},x[0].options().dtype(torch::kFloat));
    auto ids=torch::empty({b,k},x[0].options().dtype(torch::kLong));auto out=torch::empty({b,k},scores.options());
    eagle_cuda_vocab<<<dim3(b,v),256,0,stream>>>((bf16*)x[0].data_ptr(),(bf16*)x[1].data_ptr(),scores.data_ptr<float>(),h,v);
    eagle_cuda_topk<<<b,256,0,stream>>>(scores.data_ptr<float>(),ids.data_ptr<int64_t>(),out.data_ptr<float>(),(bf16*)x[0].data_ptr(),(bf16*)x[1].data_ptr(),h,v,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,out};
}
