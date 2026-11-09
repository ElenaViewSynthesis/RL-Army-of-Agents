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

__global__ void packed_varlen_reduction_kernel(float* out, const float* v, const int* off,
                                                int is_mean) {
    __shared__ float shared[32];
    int seg = blockIdx.x;
    int start = off[seg];
    int end = off[seg + 1];
    float s = 0.f;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) s += v[j];
    s = block_reduce_sum(s, shared);
    if (threadIdx.x == 0) {
        int length = end - start;
        out[seg] = is_mean ? s / fmaxf((float)length, 1.f) : s;
    }
}

extern "C" void packed_varlen_reduction_launcher(
    float* out, const float* v, const int* off, int b, int is_mean, cudaStream_t stream
) {
    if (b <= 0) return;
    packed_varlen_reduction_kernel<<<b, 256, 0, stream>>>(out, v, off, is_mean);
}
