#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

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
__global__ void masked_mean_kernel(float* out, const T* x, const int* lengths, int l) {
    __shared__ float shared[32];
    int row = blockIdx.x;
    int length = lengths[row];
    float s = 0.f;
    for (int j = threadIdx.x; j < length; j += blockDim.x) s += load_as_float(x[(long)row * l + j]);
    s = block_reduce_sum(s, shared);
    if (threadIdx.x == 0) out[row] = s / fmaxf((float)length, 1.f);
}

template<typename T>
static void masked_mean_run(float* out, const T* x, const int* lengths, int b, int l,
                            cudaStream_t stream) {
    masked_mean_kernel<T><<<b, 256, 0, stream>>>(out, x, lengths, l);
}

extern "C" void masked_mean_launcher(
    float* out, const void* x, const int* lengths, int b, int l, int input_is_bf16,
    cudaStream_t stream
) {
    if (b <= 0) return;
    if (input_is_bf16) {
        masked_mean_run<__nv_bfloat16>(out, (const __nv_bfloat16*)x, lengths, b, l, stream);
    } else {
        masked_mean_run<float>(out, (const float*)x, lengths, b, l, stream);
    }
}
