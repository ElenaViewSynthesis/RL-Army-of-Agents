#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"
extern "C" void moe_dense_gemm_launcher(float*, float*, void*, const void*, const void*, const void*, const void*, const float*, int, int, int, int, cudaStream_t);
static torch::Tensor moe_forward(torch::Tensor x, torch::Tensor gate, torch::Tensor up,
                                  torch::Tensor down, torch::Tensor alpha) {
    TORCH_CHECK(x.is_cuda() && gate.is_cuda() && up.is_cuda() && down.is_cuda() && alpha.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gate.is_contiguous() && up.is_contiguous() && down.is_contiguous() && alpha.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && gate.dim() == 3 && up.sizes() == gate.sizes() && down.dim() == 3, "requires x [T,H], gate/up [E,H,I], down [E,I,H]");
    int64_t e = gate.size(0), h = gate.size(1), i = gate.size(2);
    TORCH_CHECK(x.size(1) == h && down.size(0) == e && down.size(1) == i && down.size(2) == h, "shape mismatch");
    TORCH_CHECK(alpha.dim() == 2 && alpha.size(0) == x.size(0) && alpha.size(1) == e, "alpha must be [T,E]");
    TORCH_CHECK(x.device() == gate.device() && x.device() == up.device() && x.device() == down.device() && x.device() == alpha.device(), "inputs must share a device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && gate.scalar_type() == torch::kBFloat16 && up.scalar_type() == torch::kBFloat16 && down.scalar_type() == torch::kBFloat16 && alpha.scalar_type() == torch::kFloat, "requires bfloat16 weights, float32 alpha");
    const c10::cuda::CUDAGuard guard(x.device());
    int64_t t = x.size(0);
    auto fc1 = torch::empty({t, 2 * e * i}, x.options().dtype(torch::kFloat));
    auto act = torch::empty({t, e * i}, x.options());
    auto out = torch::empty({t, h}, x.options().dtype(torch::kFloat));
    moe_dense_gemm_launcher(out.data_ptr<float>(), fc1.data_ptr<float>(), act.data_ptr(),
                            x.data_ptr(), gate.data_ptr(), up.data_ptr(), down.data_ptr(),
                            alpha.data_ptr<float>(), (int)t, (int)h, (int)e, (int)i,
                            c10::cuda::getCurrentCUDAStream().stream());
    return out;
}
static void reg(pybind11::module& m) { m.def("moe_dense_gemm_forward", &moe_forward, py::arg("x"), py::arg("gate"), py::arg("up"), py::arg("down"), py::arg("alpha")); }
REGISTER_BINDING(moe_dense_gemm, reg);
