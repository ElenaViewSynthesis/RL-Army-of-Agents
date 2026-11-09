#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__global__ void moe_fc1_kernel(float* out, const __nv_bfloat16* x, const __nv_bfloat16* gate,
                               const __nv_bfloat16* up, int t, int h, int e, int i) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    int ei2 = 2 * e * i;
    if (col >= ei2 || row >= t) return;
    int expert = col / (2 * i);
    int which = (col / i) & 1;
    int inner = col % i;
    const __nv_bfloat16* w = which ? up : gate;
    float acc = 0.f;
    for (int kk = 0; kk < h; ++kk)
        acc += bf2f(x[(long)row * h + kk]) * bf2f(w[((long)expert * h + kk) * i + inner]);
    out[(long)row * ei2 + col] = acc;
}

__global__ void moe_act_kernel(__nv_bfloat16* out, const float* fc1, const float* alpha,
                                int t, int e, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * e * i) return;
    int ii = idx % i, e_ = (idx / i) % e, tok = idx / (i * e);
    float g = fc1[((long)tok * e + e_) * 2 * i + ii];
    float u = fc1[((long)tok * e + e_) * 2 * i + i + ii];
    out[idx] = __float2bfloat16(g / (1.f + expf(-g)) * u * alpha[(long)tok * e + e_]);
}

__global__ void moe_down_kernel(float* out, const __nv_bfloat16* act, const __nv_bfloat16* down,
                                 int t, int h, int ei) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= h || row >= t) return;
    float acc = 0.f;
    for (int kk = 0; kk < ei; ++kk)
        acc += bf2f(act[(long)row * ei + kk]) * bf2f(down[(long)kk * h + col]);
    out[(long)row * h + col] = acc;
}

extern "C" void moe_dense_gemm_launcher(
    float* out, float* fc1, void* act, const void* x, const void* gate, const void* up,
    const void* down, const float* alpha, int t, int h, int e, int i, cudaStream_t stream
) {
    if (t <= 0 || e * i <= 0) return;
    int ei2 = 2 * e * i;
    moe_fc1_kernel<<<dim3((ei2 + 255) / 256, t), 256, 0, stream>>>(
        fc1, (const __nv_bfloat16*)x, (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        t, h, e, i);
    int total = t * e * i;
    moe_act_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        (__nv_bfloat16*)act, fc1, alpha, t, e, i);
    moe_down_kernel<<<dim3((h + 255) / 256, t), 256, 0, stream>>>(
        out, (const __nv_bfloat16*)act, (const __nv_bfloat16*)down, t, h, e * i);
}
