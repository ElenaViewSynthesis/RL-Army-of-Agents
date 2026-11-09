#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

#define LOGSM_CHUNK 4096

template<typename T>
__global__ void log_softmax_chunk_kernel(float* pmax, float* psum, const T* x, int v,
                                         int chunks) {
    int row = blockIdx.x;
    int chunk = blockIdx.y;
    int start = chunk * LOGSM_CHUNK;
    if (start >= v) return;
    const T* xr = x + (long)row * v;
    __shared__ float shared[32];
    float m = -CUDART_INF_F;
    for (int j = start + threadIdx.x; j < v && j < start + LOGSM_CHUNK; j += blockDim.x)
        m = fmaxf(m, load_as_float(xr[j]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < (blockDim.x + 31) / 32 ? shared[threadIdx.x] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) shared[0] = m;
    }
    __syncthreads();
    m = shared[0];
    __syncthreads();
    float s = 0.f;
    for (int j = start + threadIdx.x; j < v && j < start + LOGSM_CHUNK; j += blockDim.x)
        s += expf(load_as_float(xr[j]) - m);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x + 31) / 32 ? shared[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (threadIdx.x == 0) {
            pmax[row * chunks + chunk] = m;
            psum[row * chunks + chunk] = s;
        }
    }
}

template<typename T>
__global__ void log_softmax_gather_kernel(float* out, const T* x, const int64_t* ids,
                                          const float* pmax, const float* psum, int v,
                                          int chunks) {
    int row = blockIdx.x;
    __shared__ float m_s, s_s;
    if (threadIdx.x == 0) {
        float m = -CUDART_INF_F;
        for (int c = 0; c < chunks; ++c) m = fmaxf(m, pmax[row * chunks + c]);
        m_s = m;
    }
    __syncthreads();
    float m = m_s;
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int c = 0; c < chunks; ++c) s += psum[row * chunks + c] * expf(pmax[row * chunks + c] - m);
        s_s = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t target = ids[row];
        float xt = (target >= 0 && target < v) ? load_as_float(x[(long)row * v + target]) : 0.f;
        out[row] = xt - m - logf(s_s);
    }
}

template<typename T>
static void log_softmax_gather_run(float* out, float* pmax, float* psum, const T* x,
                                   const int64_t* ids, int b, int v, int chunks,
                                   cudaStream_t stream) {
    dim3 grid1(b, chunks);
    log_softmax_chunk_kernel<T><<<grid1, 256, 0, stream>>>(pmax, psum, x, v, chunks);
    log_softmax_gather_kernel<T><<<b, 32, 0, stream>>>(out, x, ids, pmax, psum, v, chunks);
}

extern "C" void log_softmax_gather_launcher(
    float* out, float* pmax, float* psum, const void* x, const int64_t* ids,
    int b, int v, int input_is_bf16, cudaStream_t stream
) {
    if (b <= 0 || v <= 0) return;
    int chunks = (v + LOGSM_CHUNK - 1) / LOGSM_CHUNK;
    if (input_is_bf16) {
        log_softmax_gather_run<__nv_bfloat16>(out, pmax, psum, (const __nv_bfloat16*)x,
                                              ids, b, v, chunks, stream);
    } else {
        log_softmax_gather_run<float>(out, pmax, psum, (const float*)x, ids, b, v, chunks,
                                      stream);
    }
}
