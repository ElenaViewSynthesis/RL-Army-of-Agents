#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

__device__ __forceinline__ uint64_t topk_key(float x, int idx) {
    uint32_t u = (x == 0.f) ? 0u : __float_as_uint(x);
    uint32_t ordered = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t)ordered << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)idx);
}

__global__ void topk_chunk_kernel(uint64_t* scratch, const float* x, int v, int k, int chunks) {
    __shared__ uint64_t keys[1024];
    int row = blockIdx.x;
    int chunk = blockIdx.y;
    int i = chunk * 1024 + threadIdx.x;
    keys[threadIdx.x] = i < v ? topk_key(x[(long)row * v + i], i) : 0ull;
    __syncthreads();
    for (int size = 2; size <= 1024; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = threadIdx.x ^ stride;
            if (partner > threadIdx.x) {
                uint64_t a = keys[threadIdx.x], b = keys[partner];
                bool up = ((threadIdx.x & size) == 0);
                if ((a < b) == up) { keys[threadIdx.x] = b; keys[partner] = a; }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x < k) scratch[(long)(row * chunks + chunk) * k + threadIdx.x] = keys[threadIdx.x];
}

__global__ void topk_final_kernel(int* out, const float* x, const float* u,
                                  uint64_t* scratch, int v, int k, int chunks,
                                  float top_p) {
    __shared__ unsigned long long warp_best[8];
    __shared__ unsigned long long selected;
    __shared__ int ids[32];
    __shared__ float probs[32];
    int row = blockIdx.x;
    int total = chunks * k;
    uint64_t* cand = scratch + (long)row * total;
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
        for (int i = threadIdx.x; i < total; i += blockDim.x)
            best = cand[i] > best ? cand[i] : best;
        for (int off = 16; off > 0; off >>= 1) {
            unsigned long long other = __shfl_xor_sync(0xffffffffu, (unsigned long long)best, off);
            best = other > best ? (uint64_t)other : best;
        }
        if ((threadIdx.x & 31) == 0) warp_best[threadIdx.x >> 5] = best;
        __syncthreads();
        if (threadIdx.x < 32) {
            int warps = (blockDim.x + 31) >> 5;
            best = threadIdx.x < warps ? (uint64_t)warp_best[threadIdx.x] : 0;
            for (int off = 16; off > 0; off >>= 1) {
                unsigned long long other = __shfl_xor_sync(0xffffffffu, (unsigned long long)best, off);
                best = other > best ? (uint64_t)other : best;
            }
            if (threadIdx.x == 0) selected = best;
        }
        __syncthreads();
        best = (uint64_t)selected;
        __syncthreads();
        if (best != 0) {
            for (int i = threadIdx.x; i < total; i += blockDim.x)
                if (cand[i] == best) cand[i] = 0;
            if (threadIdx.x == 0) ids[r] = (int)(0xFFFFFFFFu - (uint32_t)(best & 0xFFFFFFFFu));
        } else if (threadIdx.x == 0) ids[r] = 0;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -CUDART_INF_F;
        for (int r = 0; r < k; ++r) mx = fmaxf(mx, x[(long)row * v + ids[r]]);
        float s = 0.f;
        for (int r = 0; r < k; ++r) { probs[r] = expf(x[(long)row * v + ids[r]] - mx); s += probs[r]; }
        float cum = 0.f, filt_sum = 0.f;
        for (int r = 0; r < k; ++r) {
            probs[r] /= s;
            bool keep = cum < top_p;
            cum += probs[r];
            if (!keep) probs[r] = 0.f;
            filt_sum += probs[r];
        }
        cum = 0.f;
        int rank = 0;
        float uni = u[row];
        for (int r = 0; r < k; ++r) {
            cum += probs[r] / filt_sum;
            if (cum <= uni) rank = r + 1;
        }
        if (rank > k - 1) rank = k - 1;
        out[row] = ids[rank];
    }
}

extern "C" void topk_sampling_launcher(
    int* out, uint64_t* scratch, const float* x, const float* u,
    int b, int v, int k, int chunks, float top_p, cudaStream_t stream
) {
    dim3 grid1(b, chunks);
    topk_chunk_kernel<<<grid1, 1024, 0, stream>>>(scratch, x, v, k, chunks);
    topk_final_kernel<<<b, 256, 0, stream>>>(out, x, u, scratch, v, k, chunks, top_p);
}
