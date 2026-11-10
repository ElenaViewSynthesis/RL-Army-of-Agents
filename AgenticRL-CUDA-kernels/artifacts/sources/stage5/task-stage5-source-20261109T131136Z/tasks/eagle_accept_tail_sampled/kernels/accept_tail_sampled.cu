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

__global__ void eagle_cuda_sample_init(const float* target,float* residual,int* winner,int total,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)residual[i]=target[i];if(i<rows)winner[i]=-1;
}
__global__ void eagle_cuda_reject(const int* tokens,const int* parents,const float* draft,float* residual,const float* uniform,int* winner,float* scratch,int n,int v,int child){
    int b=blockIdx.x,row=b*(n+1)+parents[b*n+child]+1;__shared__ int rejected;__shared__ double mass;__shared__ double partial[256];
    if(threadIdx.x==0){int token=tokens[b*n+child];float q=draft[row*v+token];float ratio=q>0?fminf(1.f,__fdiv_rn(residual[row*v+token],q)):0.f;
        bool active=winner[row]<0;rejected=active&&uniform[b*n+child]>=ratio;if(active&&!rejected)winner[row]=child;}
    __syncthreads();double total=0;
    for(int i=threadIdx.x;i<v;i+=blockDim.x){float x=fmaxf(__fsub_rn(residual[row*v+i],draft[row*v+i]),0.f);scratch[b*v+i]=x;total+=(double)x;}
    partial[threadIdx.x]=total;__syncthreads();
    for(int step=128;step;step>>=1){if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)mass=partial[0];__syncthreads();float norm=mass>0?(float)mass:1.f;
    for(int i=threadIdx.x;i<v;i+=blockDim.x){float corrected=__fdiv_rn(scratch[b*v+i],norm);if(rejected)residual[row*v+i]=corrected;}
}
__global__ void eagle_cuda_bonuses(const float* residual,const float* uniforms,int* bonus,float* probability,int v){
    int row=blockIdx.x;double cdf=0;int chosen=v-1;bool found=false;
    for(int i=0;i<v;i++){cdf+=(double)residual[row*v+i];if(!found&&cdf>(double)uniforms[row]){chosen=i;found=true;}}
    bonus[row]=chosen;probability[row]=residual[row*v+chosen];
}
__global__ void eagle_cuda_sample_walk(const int* parents,const int* slots,const int* winner,const int* bonuses,const float* probabilities,int* lengths,int* paths,int* bonus,int* compact,float* probability,int n){
    int b=blockIdx.x,current=-1,length=0;for(int i=0;i<n;i++){paths[b*n+i]=-1;compact[b*n+i]=-1;}
    for(int depth=0;depth<n;depth++){int child=winner[b*(n+1)+current+1];if(child<0)break;
        paths[b*n+length]=child;compact[b*n+length]=slots[b*n+child];length++;current=child;}
    int row=b*(n+1)+current+1;lengths[b]=length;bonus[b]=bonuses[row];probability[b]=probabilities[row];
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),n=x[0].size(1),v=x[2].size(2),rows=b*(n+1),total=rows*v;
    auto residual=torch::empty_like(x[2]);auto winner=torch::empty({b,n+1},x[0].options());auto bonuses=torch::empty_like(winner);auto probs=torch::empty({b,n+1},x[2].options());auto scratch=torch::empty({b,v},x[2].options());
    auto lengths=torch::empty({b},x[0].options());auto bonus=torch::empty_like(lengths);auto paths=torch::empty_like(x[0]);auto compact=torch::empty_like(x[4]);auto probability=torch::empty({b},x[2].options());
    eagle_cuda_sample_init<<<(total+255)/256,256,0,stream>>>(x[2].data_ptr<float>(),residual.data_ptr<float>(),winner.data_ptr<int>(),total,rows);
    for(int child=0;child<n;child++)eagle_cuda_reject<<<b,256,0,stream>>>(x[0].data_ptr<int>(),x[1].data_ptr<int>(),x[3].data_ptr<float>(),residual.data_ptr<float>(),x[5].data_ptr<float>(),winner.data_ptr<int>(),scratch.data_ptr<float>(),n,v,child);
    eagle_cuda_bonuses<<<rows,1,0,stream>>>(residual.data_ptr<float>(),x[6].data_ptr<float>(),bonuses.data_ptr<int>(),probs.data_ptr<float>(),v);
    eagle_cuda_sample_walk<<<b,1,0,stream>>>(x[1].data_ptr<int>(),x[4].data_ptr<int>(),winner.data_ptr<int>(),bonuses.data_ptr<int>(),probs.data_ptr<float>(),lengths.data_ptr<int>(),paths.data_ptr<int>(),bonus.data_ptr<int>(),compact.data_ptr<int>(),probability.data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {lengths,paths,bonus,compact,x[1],probability};
}
