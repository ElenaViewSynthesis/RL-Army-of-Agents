#include <cuda_runtime.h>

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

__global__ void grpo_group_norm_kernel(float* out, const float* x, int g, float eps) {
    __shared__ float shared[32];
    int row = blockIdx.x;
    const float* xr = x + (long)row * g;
    float* outr = out + (long)row * g;
    float s = 0.f;
    for (int j = threadIdx.x; j < g; j += blockDim.x) s += xr[j];
    float mean = block_reduce_sum(s, shared) / g;
    s = 0.f;
    for (int j = threadIdx.x; j < g; j += blockDim.x) {
        float c = xr[j] - mean;
        s += c * c;
    }
    float std = sqrtf(block_reduce_sum(s, shared) / g);
    for (int j = threadIdx.x; j < g; j += blockDim.x) outr[j] = (xr[j] - mean) / (std + eps);
}

extern "C" void grpo_group_norm_launcher(
    float* out, const float* x, int b, int g, float eps, cudaStream_t stream
) {
    if (b <= 0 || g <= 0) return;
    int threads = g < 32 ? 32 : (g < 256 ? ((g + 31) / 32) * 32 : 256);
    grpo_group_norm_kernel<<<b, threads, 0, stream>>>(out, x, g, eps);
}
