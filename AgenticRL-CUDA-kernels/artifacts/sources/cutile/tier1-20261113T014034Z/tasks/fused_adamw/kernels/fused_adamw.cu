#include <cuda_runtime.h>

__global__ void fused_adamw_kernel(float* updated, float* new_m, float* new_v,
                                   const float* p, const float* g, const float* m,
                                   const float* v, float lr, float beta1, float beta2,
                                   float eps, float weight_decay, float bc1, float bc2,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float mi = beta1 * m[i] + (1.f - beta1) * g[i];
        float vi = beta2 * v[i] + (1.f - beta2) * g[i] * g[i];
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        updated[i] = p[i] * (1.f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
        new_m[i] = mi;
        new_v[i] = vi;
    }
}

extern "C" void fused_adamw_launcher(
    float* updated, float* new_m, float* new_v, const float* p, const float* g,
    const float* m, const float* v, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2, int n, cudaStream_t stream
) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    fused_adamw_kernel<<<blocks, threads, 0, stream>>>(updated, new_m, new_v, p, g, m, v,
                                                     lr, beta1, beta2, eps, weight_decay,
                                                     bc1, bc2, n);
}
