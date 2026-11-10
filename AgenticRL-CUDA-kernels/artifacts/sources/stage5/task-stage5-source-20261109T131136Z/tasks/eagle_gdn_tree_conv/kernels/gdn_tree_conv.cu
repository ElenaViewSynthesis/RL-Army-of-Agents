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

__global__ void eagle_cuda_windows(const int* parents,int* windows,int n){
    int b=blockIdx.x,node=blockIdx.y,current=node;
    for(int i=0;i<4;i++){windows[(b*n+node)*4+i]=current;current=current>=0?parents[b*n+current]:current-1;}
}
__global__ void eagle_cuda_conv(const bf16* x,const bf16* history,const float* weight,const float* bias,const int* windows,float* out,int n,int c,bool mamba){
    int b=blockIdx.x,node=blockIdx.y,ch=blockIdx.z*blockDim.x+threadIdx.x;if(ch>=c)return;float sum=0;
    for(int i=0;i<4;i++){
        int a=windows[(b*n+node)*4+i];
        float value=__bfloat162float(a>=0?x[b*n*c+(mamba?ch*n+a:a*c+ch)]:history[(b*3+a+3)*c+ch]);
        sum=__fadd_rn(sum,__fmul_rn(value,weight[ch*4+3-i]));
    }out[b*n*c+(mamba?ch*n+node:node*c+ch)]=__fadd_rn(sum,bias[ch]);
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[4].size(0),n=x[4].size(1),c=x[2].size(0);
    auto out=torch::empty_like(x[0],x[0].options().dtype(torch::kFloat));auto windows=torch::empty({b,n,4},x[4].options());
    eagle_cuda_windows<<<dim3(b,n),1,0,stream>>>(x[4].data_ptr<int>(),windows.data_ptr<int>(),n);
    eagle_cuda_conv<<<dim3(b,n,(c+127)/128),128,0,stream>>>((bf16*)x[0].data_ptr(),(bf16*)x[1].data_ptr(),x[2].data_ptr<float>(),x[3].data_ptr<float>(),windows.data_ptr<int>(),out.data_ptr<float>(),n,c,parameter!=0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,x[4],windows};
}
