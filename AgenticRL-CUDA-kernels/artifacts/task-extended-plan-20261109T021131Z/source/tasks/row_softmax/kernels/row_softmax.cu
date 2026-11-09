#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T store_from_float(float v);
template<> __device__ __forceinline__ float store_from_float<float>(float v) { return v; }
template<> __device__ __forceinline__ __half store_from_float<__half>(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 store_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float block_reduce(float v, bool is_max, float* shared) {
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, off);
        v = is_max ? fmaxf(v, other) : v + other;
    }
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    if (threadIdx.x < 32) {
        v = threadIdx.x < warps ? shared[threadIdx.x] : (is_max ? -CUDART_INF_F : 0.f);
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, v, off);
            v = is_max ? fmaxf(v, other) : v + other;
        }
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}

template<typename T>
__global__ void row_softmax_kernel(T* out, const T* x, int h) {
    __shared__ float shared[32];
    long row = blockIdx.x;
    const T* xr = x + row * h;
    T* outr = out + row * h;
    float m = -CUDART_INF_F;
    for (int j = threadIdx.x; j < h; j += blockDim.x) m = fmaxf(m, load_as_float(xr[j]));
    m = block_reduce(m, true, shared);
    float s = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) s += expf(load_as_float(xr[j]) - m);
    s = block_reduce(s, false, shared);
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        outr[j] = store_from_float<T>(expf(load_as_float(xr[j]) - m) / s);
    }
}

template<typename T>
static void row_softmax_run(T* out, const T* x, long b, int h, cudaStream_t stream) {
    row_softmax_kernel<T><<<(int)b, 256, 0, stream>>>(out, x, h);
}

extern "C" void row_softmax_launcher(
    void* out, const void* x, long b, int h, int dtype_code, cudaStream_t stream
) {
    if (b <= 0 || h <= 0) return;
    if (dtype_code == 2) {
        row_softmax_run<__half>((__half*)out, (const __half*)x, b, h, stream);
    } else if (dtype_code == 1) {
        row_softmax_run<__nv_bfloat16>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, b, h, stream);
    } else {
        row_softmax_run<float>((float*)out, (const float*)x, b, h, stream);
    }
}
