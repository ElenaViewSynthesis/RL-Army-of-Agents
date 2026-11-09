#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__global__ void decode_scores_kernel(float* scores, const __nv_bfloat16* q,
                                     const __nv_bfloat16* k, int hq, int repeats,
                                     int s, int d, float scale) {
    int bh = blockIdx.x;
    int qh = blockIdx.y;
    int b = bh, kvh = qh / repeats;
    const __nv_bfloat16* qr = q + ((long)b * hq + qh) * d;
    const __nv_bfloat16* kr = k + ((long)b * (hq / repeats) + kvh) * s * d;
    for (int j = threadIdx.x; j < s; j += blockDim.x) {
        float acc = 0.f;
        for (int dd = 0; dd < d; ++dd) acc += bf2f(qr[dd]) * bf2f(kr[j * d + dd]);
        scores[((long)b * hq + qh) * s + j] = acc * scale;
    }
}

__global__ void decode_softmax_pv_kernel(__nv_bfloat16* out, const float* scores,
                                        const __nv_bfloat16* v, int hq, int repeats,
                                        int s, int d) {
    int b = blockIdx.x, qh = blockIdx.y;
    int kvh = qh / repeats;
    const float* sr = scores + ((long)b * hq + qh) * s;
    __shared__ float shared[32];
    float m = -CUDART_INF_F;
    for (int j = threadIdx.x; j < s; j += blockDim.x) m = fmaxf(m, sr[j]);
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
    float l = 0.f;
    for (int j = threadIdx.x; j < s; j += blockDim.x) l += expf(sr[j] - m);
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = l;
    __syncthreads();
    if (threadIdx.x < 32) {
        l = threadIdx.x < (blockDim.x + 31) / 32 ? shared[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
        if (threadIdx.x == 0) shared[0] = l;
    }
    __syncthreads();
    l = shared[0];
    const __nv_bfloat16* vr = v + ((long)b * (hq / repeats) + kvh) * s * d;
    __nv_bfloat16* outr = out + ((long)b * hq + qh) * d;
    for (int dd = threadIdx.x; dd < d; dd += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < s; ++j) acc += expf(sr[j] - m) * bf2f(vr[j * d + dd]);
        outr[dd] = __float2bfloat16(acc / l);
    }
}

extern "C" void decode_attention_launcher(
    void* out, float* scores, const void* q, const void* k, const void* v,
    int b, int hq, int hkv, int s, int d, float scale, cudaStream_t stream
) {
    dim3 grid(b, hq);
    decode_scores_kernel<<<grid, 256, 0, stream>>>(
        scores, (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, hq, hq / hkv, s, d, scale);
    decode_softmax_pv_kernel<<<grid, 256, 0, stream>>>(
        (__nv_bfloat16*)out, scores, (const __nv_bfloat16*)v, hq, hq / hkv, s, d);
}
