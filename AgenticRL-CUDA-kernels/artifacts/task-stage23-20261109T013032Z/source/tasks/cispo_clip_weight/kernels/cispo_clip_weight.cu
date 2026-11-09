#include <cuda_runtime.h>

__global__ void cispo_clip_weight_kernel(float* out, const float* lr, const float* adv,
                                          const bool* mask, float low, float high, int n, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float w = fminf(fmaxf(expf(lr[i]), low), high) * adv[i / s];
        out[i] = mask[i] ? w : 0.f;
    }
}

extern "C" void cispo_clip_weight_launcher(
    float* out, const float* lr, const float* adv, const bool* mask,
    float low, float high, int n, int s, cudaStream_t stream
) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cispo_clip_weight_kernel<<<blocks, threads, 0, stream>>>(out, lr, adv, mask, low, high, n, s);
}
