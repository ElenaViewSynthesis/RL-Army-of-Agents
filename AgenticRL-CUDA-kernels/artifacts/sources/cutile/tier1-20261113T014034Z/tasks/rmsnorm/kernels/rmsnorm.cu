#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T store_from_float(float v);
template<> __device__ __forceinline__ float store_from_float<float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_bfloat16 store_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    if (threadIdx.x < 32) {
        v = threadIdx.x < warps ? shared[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}

template<typename T>
__global__ void rmsnorm_kernel(T* out, const T* x, const T* weight, int h, float eps) {
    __shared__ float shared[32];
    long row = blockIdx.x;
    const T* xr = x + row * h;
    T* outr = out + row * h;
    float s = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        float v = load_as_float(xr[j]);
        s += v * v;
    }
    s = block_reduce_sum(s, shared);
    float inv = rsqrtf(s / h + eps);
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        outr[j] = store_from_float<T>(load_as_float(xr[j]) * inv * load_as_float(weight[j]));
    }
}

template<typename T>
static void rmsnorm_run(T* out, const T* x, const T* weight, int b, int h, float eps,
                        cudaStream_t stream) {
    rmsnorm_kernel<T><<<b, 256, 0, stream>>>(out, x, weight, h, eps);
}

extern "C" void rmsnorm_launcher(
    void* out, const void* x, const void* weight, int b, int h, float eps,
    int input_is_bf16, cudaStream_t stream
) {
    if (b <= 0 || h <= 0) return;
    if (input_is_bf16) {
        rmsnorm_run<__nv_bfloat16>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, (const __nv_bfloat16*)weight, b, h, eps, stream);
    } else {
        rmsnorm_run<float>((float*)out, (const float*)x, (const float*)weight, b, h, eps, stream);
    }
}
