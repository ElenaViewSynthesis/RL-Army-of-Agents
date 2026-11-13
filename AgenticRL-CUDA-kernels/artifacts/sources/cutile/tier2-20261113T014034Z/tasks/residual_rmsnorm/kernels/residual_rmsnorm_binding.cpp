#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"
extern "C" void residual_rmsnorm_launcher(void*, void*, const void*, const void*, const void*, int, int, float, cudaStream_t);
static pybind11::tuple rrn_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && residual.sizes() == x.sizes() && weight.numel() == x.size(1), "requires x,residual [B,H], weight [H]");
    TORCH_CHECK(x.device() == residual.device() && x.device() == weight.device(), "inputs must share a device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && residual.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16, "requires bfloat16 inputs");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x), z = torch::empty_like(x);
    residual_rmsnorm_launcher(out.data_ptr(), z.data_ptr(), x.data_ptr(), residual.data_ptr(),
                              weight.data_ptr(), (int)x.size(0), (int)x.size(1), (float)eps,
                              c10::cuda::getCurrentCUDAStream().stream());
    return pybind11::make_tuple(out, z);
}
static void reg(pybind11::module& m) { m.def("residual_rmsnorm_forward", &rrn_forward, py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps")); }
REGISTER_BINDING(residual_rmsnorm, reg);
