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

__global__ void eagle_cuda_recurrence(const float* q,const float* key,const float* value,const float* gates,const float* initial,const int* parents,float* states,float* out,int n,int heads,int dk,int dv){
    int b=blockIdx.x,head=blockIdx.y,v=blockIdx.z*blockDim.x+threadIdx.x;if(v>=dv)return;
    float state[128];
    for(int node=0;node<n;node++){
        int parent=parents[b*n+node],row=(b*n+node)*heads+head;
        float alpha=gates[row*2],beta=gates[row*2+1],projection=0;
        for(int j=0;j<dk;j++){
            float base=parent<0?initial[((b*heads+head)*dk+j)*dv+v]:states[(((b*n+parent)*heads+head)*dk+j)*dv+v];
            state[j]=__fmul_rn(base,alpha);projection=__fadd_rn(projection,__fmul_rn(key[row*dk+j],state[j]));
        }
        float delta=__fmul_rn(beta,__fsub_rn(value[row*dv+v],projection)),output=0;
        for(int j=0;j<dk;j++){
            state[j]=__fadd_rn(state[j],__fmul_rn(key[row*dk+j],delta));states[(row*dk+j)*dv+v]=state[j];
            output=__fadd_rn(output,__fmul_rn(q[row*dk+j],state[j]));
        }out[row*dv+v]=output;
    }
}
std::vector<Tensor> eagle_forward(std::vector<Tensor> x,int parameter){
    eagle_check(x);const c10::cuda::CUDAGuard guard(x[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    int b=x[0].size(0),n=x[0].size(1),h=x[0].size(2),dk=x[0].size(3),dv=x[2].size(3);TORCH_CHECK(dk<=128,"key dimension exceeds local state");
    auto paths=torch::empty({b,n,n},x[5].options());auto out=torch::empty_like(x[2]);auto states=torch::empty({b,n,h,dk,dv},x[0].options());
    eagle_cuda_paths<<<dim3(b,n),1,0,stream>>>(x[5].data_ptr<int>(),paths.data_ptr<int>(),n);
    eagle_cuda_recurrence<<<dim3(b,h,(dv+31)/32),32,0,stream>>>(x[0].data_ptr<float>(),x[1].data_ptr<float>(),x[2].data_ptr<float>(),x[3].data_ptr<float>(),x[4].data_ptr<float>(),x[5].data_ptr<int>(),states.data_ptr<float>(),out.data_ptr<float>(),n,h,dk,dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,states,x[5],paths};
}
