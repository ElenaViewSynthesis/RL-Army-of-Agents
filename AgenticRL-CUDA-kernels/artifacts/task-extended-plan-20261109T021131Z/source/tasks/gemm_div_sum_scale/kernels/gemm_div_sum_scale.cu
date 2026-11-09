#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__global__ void gemm_div_sum_matmul_kernel(float* product, const T* a, const T* b,
                                           int m, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < n) {
        float acc = 0.f;
        for (int kk = 0; kk < k; ++kk) {
            acc += load_as_float(a[i * k + kk]) * load_as_float(b[kk * n + j]);
        }
        product[i * n + j] = acc;
    }
}

__global__ void gemm_div_sum_rowsum_kernel(float* out, const float* product,
                                           int m, int n, float divisor, float scale) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    if (row >= m) return;
    float acc = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) acc += product[row * n + j];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) shared[threadIdx.x] += shared[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = (shared[0] / divisor) * scale;
}

template<typename T>
static void gemm_div_sum_run(float* out, float* product, const T* a, const T* b,
                             int m, int n, int k, float divisor, float scale,
                             cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (m + 15) / 16);
    gemm_div_sum_matmul_kernel<T><<<blocks, threads, 0, stream>>>(product, a, b, m, n, k);
    gemm_div_sum_rowsum_kernel<<<m, 256, 0, stream>>>(out, product, m, n, divisor, scale);
}

extern "C" void gemm_div_sum_scale_launcher(
    float* out, float* product, const void* a, const void* b,
    int m, int n, int k, float divisor, float scale, int input_is_bf16,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0) return;
    if (input_is_bf16) {
        gemm_div_sum_run<__nv_bfloat16>(out, product, (const __nv_bfloat16*)a,
                                        (const __nv_bfloat16*)b, m, n, k, divisor, scale, stream);
    } else {
        gemm_div_sum_run<float>(out, product, (const float*)a, (const float*)b,
                                m, n, k, divisor, scale, stream);
    }
}
