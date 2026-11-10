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

__global__ void eagle_cuda_attention(const bf16* q,const bf16* k,const bf16* v,const bf16* tk,const bf16* tv,const int* paths,float* part,float* maxima,float* denominators,int n,int s,int hq,int hk,int d,int window,int splits){
    int bh=blockIdx.x,b=bh/hq,head=bh%hq,node=blockIdx.y,split=blockIdx.z,ch=threadIdx.x,kvhead=head/(hq/hk);
    float query=__bfloat162float(q[((b*hq+head)*n+node)*d+ch]);
    float maximum=-INFINITY,denom=0,weighted=0;
    for(int ix=split*256;ix<min((split+1)*256,window+n);ix++){
        bool committed=ix<window;int tid=ix-window;bool visible=committed;
        if(!committed)for(int z=0;z<n;z++)visible|=paths[(b*n+node)*n+z]==tid;
        if(!visible)continue;
        long long address=committed?((long long)(b*hk+kvhead)*s+s-window+ix)*d+ch:((long long)(b*hk+kvhead)*n+tid)*d+ch;
        float key=__bfloat162float(committed?k[address]:tk[address]);
        float value=__bfloat162float(committed?v[address]:tv[address]);
        float score=eagle_reduce(query*key)*rsqrtf((float)d);
        float next=fmaxf(maximum,score),scale=expf(maximum-next),prob=expf(score-next);
        weighted=weighted*scale+prob*value;denom=denom*scale+prob;maximum=next;
    }
    long long row=((long long)bh*n+node)*splits+split;  // *d exceeds 2**31 at 32k contexts
    part[row*d+ch]=weighted;if(ch==0){maxima[row]=maximum;denominators[row]=denom;}
}
__global__ void eagle_cuda_attention_merge(const float* part,const float* maxima,const float* denominators,bf16* out,int d,int splits){
    long long row=blockIdx.x;int ch=threadIdx.x;float maximum=-INFINITY;
    for(int i=0;i<splits;i++)maximum=fmaxf(maximum,maxima[row*splits+i]);
    float sum=0,denom=0;for(int i=0;i<splits;i++){
        long long offset=row*splits+i;float scale=expf(maxima[offset]-maximum);
        sum+=part[offset*d+ch]*scale;denom+=denominators[offset]*scale;
    }out[row*d+ch]=__float2bfloat16_rn(sum/denom);
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int window){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),hq=x[0].size(1),n=x[0].size(2),d=x[0].size(3),hk=x[1].size(1),s=x[1].size(2),splits=(window+n+255)/256;
    auto paths=torch::empty({b,n,n},x[5].options());auto out=torch::empty_like(x[0]);
    auto part=torch::empty({b*hq*n,splits,d},x[0].options().dtype(torch::kFloat));auto maxima=torch::empty({b*hq*n,splits},part.options());auto denom=torch::empty_like(maxima);
    eagle_cuda_paths<<<dim3(b,n),1,0,stream>>>(x[5].data_ptr<int>(),paths.data_ptr<int>(),n);
    eagle_cuda_attention<<<dim3(b*hq,n,splits),d,0,stream>>>((bf16*)x[0].data_ptr(),(bf16*)x[1].data_ptr(),(bf16*)x[2].data_ptr(),(bf16*)x[3].data_ptr(),(bf16*)x[4].data_ptr(),paths.data_ptr<int>(),part.data_ptr<float>(),maxima.data_ptr<float>(),denom.data_ptr<float>(),n,s,hq,hk,d,window,splits);
    eagle_cuda_attention_merge<<<b*hq*n,d,0,stream>>>(part.data_ptr<float>(),maxima.data_ptr<float>(),denom.data_ptr<float>(),(bf16*)out.data_ptr(),d,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,paths};
}
