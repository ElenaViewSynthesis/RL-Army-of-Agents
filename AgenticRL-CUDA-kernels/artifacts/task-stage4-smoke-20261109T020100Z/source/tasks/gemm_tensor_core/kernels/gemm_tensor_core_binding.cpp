#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void gemm_tensor_core_launcher(
    float* out, const void* a, const void* b, const float* bias,
    int m, int n, int k, int input_is_fp16, cudaStream_t stream
);

static torch::Tensor gemm_tensor_core_forward(torch::Tensor a, torch::Tensor b,
                                                torch::Tensor bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && bias.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && bias.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "requires a [M,K] and b [K,N]");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == b.size(1) &&
                bias.scalar_type() == torch::kFloat, "bias must be float32 [N]");
    TORCH_CHECK(a.device() == b.device() && a.device() == bias.device(),
                "inputs must be on the same device");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtypes must match");
    TORCH_CHECK(a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
                "requires float16 or bfloat16 inputs");
    const c10::cuda::CUDAGuard guard(a.device());
    int64_t m = a.size(0), k = a.size(1), n = b.size(1);
    auto out = torch::empty({m, n}, a.options().dtype(torch::kFloat));
    gemm_tensor_core_launcher(out.data_ptr<float>(), a.data_ptr(), b.data_ptr(),
                              bias.data_ptr<float>(), (int)m, (int)n, (int)k,
                              a.scalar_type() == torch::kHalf ? 1 : 0,
                              c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_gemm_tensor_core(pybind11::module& m) {
    m.def("gemm_tensor_core_forward", &gemm_tensor_core_forward,
          py::arg("a"), py::arg("b"), py::arg("bias"));
}

REGISTER_BINDING(gemm_tensor_core, register_gemm_tensor_core);
