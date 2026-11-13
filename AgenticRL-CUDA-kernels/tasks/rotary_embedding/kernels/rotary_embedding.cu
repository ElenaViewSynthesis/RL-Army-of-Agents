#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T store_from_float(float v);
template<> __device__ __forceinline__ float store_from_float<float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_bfloat16 store_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void rotary_embedding_cuda_kernel(T* out, const T* x, const float* cos_table,
                                             const float* sin_table, long n_rows, long seq, long half) {
    long n = n_rows * half;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        long row = i / half, j = i - row * half;
        long t = (row % seq) * half + j;
        float c = cos_table[t], s = sin_table[t];
        float x1 = load_as_float(x[row * 2 * half + j]);
        float x2 = load_as_float(x[row * 2 * half + half + j]);
        out[row * 2 * half + j] = store_from_float<T>(x1 * c - x2 * s);
        out[row * 2 * half + half + j] = store_from_float<T>(x2 * c + x1 * s);
    }
}

template<typename T>
static void rotary_embedding_run(T* out, const T* x, const float* cos_table, const float* sin_table,
                                 long n_rows, long seq, long half, cudaStream_t stream) {
    long blocks = (n_rows * half + 255) / 256;
    if (blocks > 1048576) blocks = 1048576;  // grid-stride loop covers the rest
    rotary_embedding_cuda_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>(
        out, x, cos_table, sin_table, n_rows, seq, half);
}

extern "C" void rotary_embedding_launcher(void* out, const void* x, const void* cos_table,
                                          const void* sin_table, long n_rows, long seq, long half,
                                          int input_is_bf16, cudaStream_t stream) {
    if (n_rows <= 0 || half <= 0) return;
    if (input_is_bf16) {
        rotary_embedding_run<__nv_bfloat16>((__nv_bfloat16*)out, (const __nv_bfloat16*)x,
                                            (const float*)cos_table, (const float*)sin_table,
                                            n_rows, seq, half, stream);
    } else {
        rotary_embedding_run<float>((float*)out, (const float*)x, (const float*)cos_table,
                                    (const float*)sin_table, n_rows, seq, half, stream);
    }
}
