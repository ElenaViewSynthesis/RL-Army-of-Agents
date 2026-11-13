#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T store_from_float(float v);
template<> __device__ __forceinline__ float store_from_float<float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_bfloat16 store_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void silu_and_mul_cuda_kernel(T* out, const T* x, long rows, long width) {
    long n = rows * width;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        long row = i / width, col = i - row * width;
        float gate = load_as_float(x[row * 2 * width + col]);
        float up = load_as_float(x[row * 2 * width + width + col]);
        out[i] = store_from_float<T>(gate / (1.f + expf(-gate)) * up);
    }
}

template<typename T>
static void silu_and_mul_run(T* out, const T* x, long rows, long width, cudaStream_t stream) {
    long n = rows * width;
    long blocks = (n + 255) / 256;
    if (blocks > 1048576) blocks = 1048576;  // grid-stride loop covers the rest
    silu_and_mul_cuda_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>(out, x, rows, width);
}

extern "C" void silu_and_mul_launcher(void* out, const void* x, long rows, long width, int input_is_bf16,
                                cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    if (input_is_bf16) {
        silu_and_mul_run<__nv_bfloat16>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, rows, width, stream);
    } else {
        silu_and_mul_run<float>((float*)out, (const float*)x, rows, width, stream);
    }
}
