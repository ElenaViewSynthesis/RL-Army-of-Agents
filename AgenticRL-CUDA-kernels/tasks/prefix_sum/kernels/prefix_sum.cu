#include <cuda_runtime.h>

#define SCAN_BLOCK 1024

template<typename T>
__global__ void prefix_scan_block_kernel(T* out, const T* x, T* totals, int n) {
    __shared__ T shared[SCAN_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * SCAN_BLOCK + tid;
    shared[tid] = i < n ? x[i] : (T)0;
    __syncthreads();
    for (int off = 1; off < SCAN_BLOCK; off <<= 1) {
        T v = tid >= off ? shared[tid - off] : (T)0;
        __syncthreads();
        shared[tid] = shared[tid] + v;
        __syncthreads();
    }
    if (i < n) out[i] = shared[tid];
    if (tid == SCAN_BLOCK - 1) totals[blockIdx.x] = shared[tid];
}

template<typename T>
__global__ void prefix_scan_offset_kernel(T* out, const T* totals, int n) {
    int i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    if (blockIdx.x > 0 && i < n) out[i] = out[i] + totals[blockIdx.x - 1];
}

template<typename T>
static void prefix_scan_device(T* data, int n, cudaStream_t stream) {
    int blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    T* totals = nullptr;
    cudaMallocAsync(&totals, blocks * sizeof(T), stream);
    prefix_scan_block_kernel<T><<<blocks, SCAN_BLOCK, 0, stream>>>(data, data, totals, n);
    if (blocks > 1) {
        prefix_scan_device<T>(totals, blocks, stream);
        prefix_scan_offset_kernel<T><<<blocks, SCAN_BLOCK, 0, stream>>>(data, totals, n);
    }
    cudaFreeAsync(totals, stream);
}

extern "C" void prefix_sum_launcher(
    void* out, const void* x, int n, int input_is_int32, cudaStream_t stream
) {
    if (n <= 0) return;
    if (input_is_int32) {
        cudaMemcpyAsync(out, x, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        prefix_scan_device<int>((int*)out, n, stream);
    } else {
        cudaMemcpyAsync(out, x, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        prefix_scan_device<float>((float*)out, n, stream);
    }
}
