#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

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
    float r = shared[0];
    __syncthreads();
    return r;
}

__global__ void residual_rmsnorm_kernel(__nv_bfloat16* out, __nv_bfloat16* z,
                                         const __nv_bfloat16* x, const __nv_bfloat16* residual,
                                         const __nv_bfloat16* weight, int h, float eps) {
    __shared__ float shared[32];
    long row = blockIdx.x;
    float s = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        __nv_bfloat16 zv = __float2bfloat16(bf2f(x[row * h + j]) + bf2f(residual[row * h + j]));
        z[row * h + j] = zv;
        float f = bf2f(zv);
        s += f * f;
    }
    float inv = rsqrtf(block_reduce_sum(s, shared) / h + eps);
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        out[row * h + j] = __float2bfloat16(bf2f(z[row * h + j]) * inv * bf2f(weight[j]));
    }
}

extern "C" void residual_rmsnorm_launcher(
    void* out, void* z, const void* x, const void* residual, const void* weight,
    int b, int h, float eps, cudaStream_t stream
) {
    if (b <= 0 || h <= 0) return;
    residual_rmsnorm_kernel<<<b, 256, 0, stream>>>(
        (__nv_bfloat16*)out, (__nv_bfloat16*)z, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)residual, (const __nv_bfloat16*)weight, h, eps);
}
