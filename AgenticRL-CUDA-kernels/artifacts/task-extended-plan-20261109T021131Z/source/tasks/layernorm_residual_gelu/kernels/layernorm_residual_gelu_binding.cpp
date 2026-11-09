#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void layernorm_residual_gelu_launcher(
    void* out, const void* x, const void* residual, const void* weight, const void* bias,
    int b, int h, float eps, int input_is_bf16, cudaStream_t stream
);

static torch::Tensor layernorm_residual_gelu_forward(torch::Tensor x, torch::Tensor residual,
                                                     torch::Tensor weight, torch::Tensor bias,
                                                     double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous() &&
                bias.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2, "requires x [B,H]");
    TORCH_CHECK(residual.sizes() == x.sizes() && weight.numel() == x.size(1) &&
                bias.numel() == x.size(1), "shape mismatch against x [B,H], weight/bias [H]");
    TORCH_CHECK(x.device() == residual.device() && x.device() == weight.device() &&
                x.device() == bias.device(), "inputs must be on the same device");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "inputs must share x's dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 input");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    layernorm_residual_gelu_launcher(out.data_ptr(), x.data_ptr(), residual.data_ptr(),
                                     weight.data_ptr(), bias.data_ptr(),
                                     (int)x.size(0), (int)x.size(1), (float)eps,
                                     x.scalar_type() == torch::kBFloat16 ? 1 : 0,
                                     c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_layernorm_residual_gelu(pybind11::module& m) {
    m.def("layernorm_residual_gelu_forward", &layernorm_residual_gelu_forward,
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
}

REGISTER_BINDING(layernorm_residual_gelu, register_layernorm_residual_gelu);
