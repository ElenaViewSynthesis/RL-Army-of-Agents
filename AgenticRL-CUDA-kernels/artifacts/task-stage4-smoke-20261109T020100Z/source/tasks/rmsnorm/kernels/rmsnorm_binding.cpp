#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void rmsnorm_launcher(
    void* out, const void* x, const void* weight, int b, int h, float eps,
    int input_is_bf16, cudaStream_t stream
);

static torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1 && weight.size(0) == x.size(1),
                "requires x [B,H] and weight [H]");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight must match x dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 input");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    rmsnorm_launcher(out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                     (int)x.size(0), (int)x.size(1), (float)eps,
                     x.scalar_type() == torch::kBFloat16 ? 1 : 0,
                     c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_rmsnorm(pybind11::module& m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, py::arg("x"), py::arg("weight"), py::arg("eps"));
}

REGISTER_BINDING(rmsnorm, register_rmsnorm);
