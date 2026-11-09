#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

__device__ __forceinline__ float load_as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

__global__ void fa_scores_kernel(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* k,
                                 int s, int d, float scale, int causal) {
    __shared__ float sq[64][64];
    __shared__ float sk[64][64];
    int row = blockIdx.y * 64 + threadIdx.x;
    int bh = blockIdx.z;
    const __nv_bfloat16* qr = q + (long)bh * s * d;
    const __nv_bfloat16* kr = k + (long)bh * s * d;
    int n_start = 0, n_end = s;
    for (int n0 = n_start; n0 < n_end; n0 += 64) {
        for (int j = threadIdx.x; j < 64 * d; j += blockDim.x) {
            int ii = j / d, dd = j % d;
            int qi = blockIdx.y * 64 + ii, ki = n0 + ii;
            sq[ii][dd] = qi < s ? load_as_float(qr[qi * d + dd]) : 0.f;
            sk[ii][dd] = ki < s ? load_as_float(kr[ki * d + dd]) : 0.f;
        }
        __syncthreads();
        for (int j = blockIdx.y * 64 + threadIdx.y; j < min(s, (int)blockIdx.y * 64 + 64); j += blockDim.y) {
            for (int jj = n0 + threadIdx.x; jj < n0 + 64 && jj < s; jj += blockDim.x) {
                float acc = 0.f;
                for (int dd = 0; dd < d; ++dd) acc += sq[j - blockIdx.y * 64][dd] * sk[jj - n0][dd];
                float val = acc * scale;
                if (causal && jj > j) val = -CUDART_INF_F;
                scores[((long)bh * s + j) * s + jj] = val;
            }
        }
        __syncthreads();
    }
}

__global__ void fa_softmax_pv_kernel(__nv_bfloat16* out, const float* scores,
                                    const __nv_bfloat16* v, int s, int d, int causal) {
    int row = blockIdx.x;
    int bh = blockIdx.y;
    const float* sr = scores + ((long)bh * s + row) * s;
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
    const __nv_bfloat16* vr = v + (long)bh * s * d;
    __nv_bfloat16* outr = out + ((long)bh * s + row) * d;
    for (int dd = threadIdx.x; dd < d; dd += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < s; ++j) acc += expf(sr[j] - m) * load_as_float(vr[j * d + dd]);
        outr[dd] = __float2bfloat16(acc / l);
    }
}

extern "C" void fa_launcher(void* out, float* scores, const void* q, const void* k,
                            const void* v, int bh, int s, int d, float scale,
                            int causal, cudaStream_t stream) {
    dim3 grid1(1, (s + 63) / 64, bh);
    fa_scores_kernel<<<grid1, dim3(64, 1), 0, stream>>>(
        scores, (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, s, d, scale, causal);
    dim3 grid2(s, bh);
    fa_softmax_pv_kernel<<<grid2, 256, 0, stream>>>(
        (__nv_bfloat16*)out, scores, (const __nv_bfloat16*)v, s, d, causal);
}
