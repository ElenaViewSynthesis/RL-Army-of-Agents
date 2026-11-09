#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int GEMM_TC_BM = 64;
constexpr int GEMM_TC_BN = 64;
constexpr int GEMM_TC_BK = 32;
constexpr int GEMM_TC_THREADS = 128;

template<typename T> struct WmmaType;
template<> struct WmmaType<__half> { using type = __half; };
template<> struct WmmaType<__nv_bfloat16> { using type = __nv_bfloat16; };

template<typename T> __device__ __forceinline__ T zero_value();
template<> __device__ __forceinline__ __half zero_value<__half>() { return __float2half_rn(0.f); }
template<> __device__ __forceinline__ __nv_bfloat16 zero_value<__nv_bfloat16>() { return __float2bfloat16(0.f); }

template<typename T>
__global__ void gemm_tensor_core_kernel(float* out, const T* a, const T* b,
                                        const float* bias, int m, int n, int k) {
    __shared__ __align__(32) T As[2][GEMM_TC_BM][GEMM_TC_BK];
    __shared__ __align__(32) T Bs[2][GEMM_TC_BN][GEMM_TC_BK];
    __shared__ __align__(32) float Cs[GEMM_TC_BM][GEMM_TC_BN];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int warp_m = (warp >> 1) * 32;
    int warp_n = (warp & 1) * 32;
    int block_m = blockIdx.y * GEMM_TC_BM;
    int block_n = blockIdx.x * GEMM_TC_BN;
    int ktiles = (k + GEMM_TC_BK - 1) / GEMM_TC_BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    auto stage = [&](int buf, int k0) {
        for (int idx = tid; idx < GEMM_TC_BM * GEMM_TC_BK; idx += GEMM_TC_THREADS) {
            int i = idx / GEMM_TC_BK, kk = idx % GEMM_TC_BK;
            int gi = block_m + i, gk = k0 + kk;
            As[buf][i][kk] = (gi < m && gk < k) ? a[gi * k + gk] : zero_value<T>();
        }
        for (int idx = tid; idx < GEMM_TC_BN * GEMM_TC_BK; idx += GEMM_TC_THREADS) {
            int j = idx / GEMM_TC_BK, kk = idx % GEMM_TC_BK;
            int gj = block_n + j, gk = k0 + kk;
            Bs[buf][j][kk] = (gj < n && gk < k) ? b[gk * n + gj] : zero_value<T>();
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int kt = 0; kt < ktiles; ++kt) {
        int buf = kt & 1;
        if (kt + 1 < ktiles) stage(buf ^ 1, (kt + 1) * GEMM_TC_BK);
        for (int kk = 0; kk < GEMM_TC_BK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, typename WmmaType<T>::type,
                           wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, typename WmmaType<T>::type,
                           wmma::col_major> bf[2];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[buf][warp_m + i * 16][kk * 16], GEMM_TC_BK);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[buf][warp_n + j * 16][kk * 16], GEMM_TC_BK);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[warp_m + i * 16][warp_n + j * 16], acc[i][j],
                                    GEMM_TC_BN, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < GEMM_TC_BM * GEMM_TC_BN; idx += GEMM_TC_THREADS) {
        int i = idx / GEMM_TC_BN, j = idx % GEMM_TC_BN;
        int gi = block_m + i, gj = block_n + j;
        if (gi < m && gj < n) out[gi * n + gj] = Cs[i][j] + bias[gj];
    }
}

template<typename T>
static void gemm_tensor_core_run(float* out, const T* a, const T* b, const float* bias,
                                 int m, int n, int k, cudaStream_t stream) {
    dim3 blocks((n + GEMM_TC_BN - 1) / GEMM_TC_BN, (m + GEMM_TC_BM - 1) / GEMM_TC_BM);
    gemm_tensor_core_kernel<T><<<blocks, GEMM_TC_THREADS, 0, stream>>>(out, a, b, bias, m, n, k);
}

extern "C" void gemm_tensor_core_launcher(
    float* out, const void* a, const void* b, const float* bias,
    int m, int n, int k, int input_is_fp16, cudaStream_t stream
) {
    if (m <= 0 || n <= 0) return;
    if (input_is_fp16) {
        gemm_tensor_core_run<__half>(out, (const __half*)a, (const __half*)b, bias,
                                     m, n, k, stream);
    } else {
        gemm_tensor_core_run<__nv_bfloat16>(out, (const __nv_bfloat16*)a,
                                            (const __nv_bfloat16*)b, bias, m, n, k, stream);
    }
}
