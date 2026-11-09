#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void gemm_div_sum_scale_launcher(
    float* out, float* product, const void* a, const void* b,
    int m, int n, int k, float divisor, float scale, int input_is_bf16,
    cudaStream_t stream
);

static torch::Tensor gemm_div_sum_scale_forward(torch::Tensor a, torch::Tensor b,
                                                double divisor, double scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "requires a [M,K] and b [K,N]");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtypes must match");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat || a.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 inputs");
    const c10::cuda::CUDAGuard guard(a.device());
    int64_t m = a.size(0), k = a.size(1), n = b.size(1);
    auto product = torch::empty({m, n}, a.options().dtype(torch::kFloat));
    auto out = torch::empty({m, 1}, a.options().dtype(torch::kFloat));
    gemm_div_sum_scale_launcher(
        out.data_ptr<float>(), product.data_ptr<float>(), a.data_ptr(), b.data_ptr(),
        (int)m, (int)n, (int)k, (float)divisor, (float)scale,
        a.scalar_type() == torch::kBFloat16 ? 1 : 0,
        c10::cuda::getCurrentCUDAStream().stream()
    );
    return out;
}

static void register_gemm_div_sum_scale(pybind11::module& m) {
    m.def("gemm_div_sum_scale_forward", &gemm_div_sum_scale_forward,
          py::arg("a"), py::arg("b"), py::arg("divisor"), py::arg("scale"));
}

REGISTER_BINDING(gemm_div_sum_scale, register_gemm_div_sum_scale);
