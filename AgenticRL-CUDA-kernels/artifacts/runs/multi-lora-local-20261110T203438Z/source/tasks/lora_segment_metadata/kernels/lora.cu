// Original standalone multi-LoRA implementation; no upstream kernel code copied.
#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using Tensor = torch::Tensor;

// Native CUDA storage types keep device symbols independent of Torch namespaces.
// Explicit intrinsics also work when Torch disables implicit half conversions.
template<class T> __device__ float lora_read(T x);
template<> __device__ float lora_read(__half x){return __half2float(x);}
template<> __device__ float lora_read(__nv_bfloat16 x){return __bfloat162float(x);}
template<class T> __device__ T lora_write(float x);
template<> __device__ __half lora_write<__half>(float x){return __float2half_rn(x);}
template<> __device__ __nv_bfloat16 lora_write<__nv_bfloat16>(float x){return __float2bfloat16_rn(x);}

__global__ void lora_cuda_metadata(const int* idx,int* order,int* ids,int* starts,int* lengths,int* active,int t,int l) {
    // One block per possible slot; stable placement without host reads.
    int slot=blockIdx.x;
    int start=0, segment=0, length=0;
    for(int i=0;i<t;i++) {int id=idx[i]<0?l:idx[i];start+=id<slot;length+=id==slot;}
    for(int j=0;j<slot;j++){bool found=false;for(int i=0;i<t;i++)found|=idx[i]==j;segment+=found;}
    if(threadIdx.x==0 && slot<l && length){ids[segment]=slot;starts[segment]=start;lengths[segment]=length;}
    if(slot==l && threadIdx.x==0){
        active[0]=segment;
        for(int j=segment;j<l;j++){ids[j]=-1;starts[j]=0;lengths[j]=0;}
    }
    for(int i=threadIdx.x;i<t;i+=blockDim.x){
        int id=idx[i]<0?l:idx[i];
        if(id==slot){int offset=0;for(int j=0;j<i;j++)offset+=(idx[j]<0?l:idx[j])==slot;order[start+offset]=i;}
    }
}

template<class scalar_t> __global__ void lora_cuda_copy(const scalar_t* src,scalar_t* dst,int64_t count){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)dst[i]=src[i];
}

template<class scalar_t> __global__ void lora_cuda_shrink(const scalar_t* x,const scalar_t* a,const int* idx,const int* ranks,const float* scales,float* out,
    const int* order,const int* ids,const int* starts,const int* lengths,const int* active,int64_t t,int64_t h,int64_t r,int64_t s,int64_t l,bool segmented){
    int64_t row=blockIdx.x,rank=row%r,token=row/r%t,sl=row/(r*t);
    int slot=idx[token];
    if(segmented){int seg=blockIdx.y; if(seg>=active[0]||token>=lengths[seg])return;slot=ids[seg];token=order[starts[seg]+token];}
    float value=0;
    if(slot>=0 && rank<ranks[slot])for(int64_t k=threadIdx.x;k<h;k+=blockDim.x)value+=lora_read(x[token*h+k])*lora_read(a[((int64_t(slot)*s+sl)*r+rank)*h+k]);
    __shared__ float sums[256];sums[threadIdx.x]=value;__syncthreads();
    for(int step=128;step;step>>=1){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)out[(sl*t+token)*r+rank]=slot>=0?sums[0]*scales[slot]:0.f;
}

template<class scalar_t> __global__ void lora_cuda_expand(const float* buf,const scalar_t* b,const scalar_t* y,const int* idx,const int* ranks,const int* offsets,scalar_t* out,
    const int* order,const int* ids,const int* starts,const int* lengths,const int* active,int64_t t,int64_t n,int64_t r,int64_t s,int64_t width,bool segmented){
    int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=s*t*n)return;
    int64_t col=index%n,token=index/n%t,sl=index/(n*t);int slot=idx[token];
    if(segmented){int seg=blockIdx.y;if(seg>=active[0]||token>=lengths[seg])return;slot=ids[seg];token=order[starts[seg]+token];}
    if(slot<0)return;
    float sum=0;for(int k=0;k<ranks[slot];k++)sum+=buf[(sl*t+token)*r+k]*lora_read(b[((int64_t(slot)*s+sl)*n+col)*r+k]);
    int64_t dest=token*width+offsets[sl]+col;out[dest]=lora_write<scalar_t>(lora_read(y[dest])+sum);
}

template<class scalar_t> void lora_launch_shrink(const std::vector<Tensor>& args,Tensor& out,
    const int* order,const int* ids,const int* starts,const int* lengths,const int* active,
    int64_t t,int64_t h,int64_t r,int64_t s,int64_t l,bool segmented,cudaStream_t stream){
    lora_cuda_shrink<scalar_t><<<dim3(s*t*r,segmented?l:1),256,0,stream>>>(
        static_cast<const scalar_t*>(args[0].data_ptr()),static_cast<const scalar_t*>(args[1].data_ptr()),
        args[2].data_ptr<int>(),args[3].data_ptr<int>(),args[4].data_ptr<float>(),out.data_ptr<float>(),
        order,ids,starts,lengths,active,t,h,r,s,l,segmented);
}
template<class scalar_t> void lora_launch_expand(const std::vector<Tensor>& args,Tensor& out,
    const int* order,const int* ids,const int* starts,const int* lengths,const int* active,
    int64_t t,int64_t n,int64_t r,int64_t s,int64_t l,int64_t width,bool segmented,cudaStream_t stream){
    auto y=static_cast<const scalar_t*>(args[2].data_ptr());auto dest=static_cast<scalar_t*>(out.data_ptr());
    lora_cuda_copy<scalar_t><<<(out.numel()+255)/256,256,0,stream>>>(y,dest,out.numel());
    lora_cuda_expand<scalar_t><<<dim3((s*t*n+127)/128,segmented?l:1),128,0,stream>>>(
        args[0].data_ptr<float>(),static_cast<const scalar_t*>(args[1].data_ptr()),y,
        args[3].data_ptr<int>(),args[4].data_ptr<int>(),args[5].data_ptr<int>(),dest,
        order,ids,starts,lengths,active,t,n,r,s,width,segmented);
}

std::vector<Tensor> lora_forward(std::vector<Tensor> args,int op,bool segmented){
    for(auto &x:args)TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.device()==args[0].device(),"contiguous same-device CUDA inputs required");
    const c10::cuda::CUDAGuard guard(args[0].device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
    if(op==2){
        auto idx=args[0];int t=idx.numel(),l=args[1].numel();auto order=torch::empty_like(idx);
        auto ids=torch::empty({l},idx.options()),starts=torch::empty_like(ids),lengths=torch::empty_like(ids),active=torch::empty({1},idx.options());
        lora_cuda_metadata<<<l+1,128,0,stream>>>(idx.data_ptr<int>(),order.data_ptr<int>(),ids.data_ptr<int>(),starts.data_ptr<int>(),lengths.data_ptr<int>(),active.data_ptr<int>(),t,l);
        C10_CUDA_KERNEL_LAUNCH_CHECK();return {order,ids,starts,lengths,active};
    }
    if(op==3){
        std::vector<Tensor> sh={args[0],args[1],args[4],args[5],args[6]};
        if(segmented)sh.insert(sh.end(),args.begin()+8,args.end());
        auto buf=lora_forward(sh,0,segmented)[0];
        std::vector<Tensor> ex={buf,args[2],args[3],args[4],args[5],args[7]};
        if(segmented)ex.insert(ex.end(),args.begin()+8,args.end());
        return lora_forward(ex,1,segmented);
    }
    bool shrink=op==0;auto weight=args[1];int64_t l=weight.size(0),s=weight.size(1),r=shrink?weight.size(2):weight.size(3);
    int64_t t=shrink?args[0].size(0):args[0].size(1),h=weight.size(3),n=weight.size(2);
    auto out=shrink?torch::empty({s,t,r},args[0].options().dtype(torch::kFloat32)):torch::empty_like(args[2]);
    int meta=shrink?5:6;const int *order=nullptr,*ids=nullptr,*starts=nullptr,*lengths=nullptr,*active=nullptr;
    if(segmented){order=args[meta].data_ptr<int>();ids=args[meta+1].data_ptr<int>();starts=args[meta+2].data_ptr<int>();lengths=args[meta+3].data_ptr<int>();active=args[meta+4].data_ptr<int>();}
    if(shrink){
        // Initialize skipped rows and padding before segment programs write active rows.
        if(segmented)C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(),0,out.numel()*4,stream));
        if(args[0].scalar_type()==at::kHalf)
            lora_launch_shrink<__half>(args,out,order,ids,starts,lengths,active,t,h,r,s,l,segmented,stream);
        else {
            TORCH_CHECK(args[0].scalar_type()==at::kBFloat16,"fp16 or bf16 storage required");
            lora_launch_shrink<__nv_bfloat16>(args,out,order,ids,starts,lengths,active,t,h,r,s,l,segmented,stream);
        }
    }else{
        int64_t width=args[2].size(1);
        if(args[2].scalar_type()==at::kHalf)
            lora_launch_expand<__half>(args,out,order,ids,starts,lengths,active,t,n,r,s,l,width,segmented,stream);
        else {
            TORCH_CHECK(args[2].scalar_type()==at::kBFloat16,"fp16 or bf16 storage required");
            lora_launch_expand<__nv_bfloat16>(args,out,order,ids,starts,lengths,active,t,n,r,s,l,width,segmented,stream);
        }
    }C10_CUDA_KERNEL_LAUNCH_CHECK();return {out};
}
