#include <cmath>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void fused_adamw_launcher(
    float* updated, float* new_m, float* new_v, const float* p, const float* g,
    const float* m, const float* v, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2, int n, cudaStream_t stream
);

static pybind11::tuple fused_adamw_forward(torch::Tensor p, torch::Tensor g, torch::Tensor m,
                                           torch::Tensor v, double lr, double beta1,
                                           double beta2, double eps, double weight_decay,
                                           double step) {
    TORCH_CHECK(p.is_cuda() && g.is_cuda() && m.is_cuda() && v.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous() && v.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(p.sizes() == g.sizes() && p.sizes() == m.sizes() && p.sizes() == v.sizes(),
                "inputs must have the same shape");
    TORCH_CHECK(p.device() == g.device() && p.device() == m.device() && p.device() == v.device(),
                "inputs must be on the same device");
    TORCH_CHECK(p.scalar_type() == torch::kFloat, "inputs must be float32");
    const c10::cuda::CUDAGuard guard(p.device());
    auto updated = torch::empty_like(p);
    auto new_m = torch::empty_like(m);
    auto new_v = torch::empty_like(v);
    float bc1 = (float)(1.0 - std::pow(beta1, step));
    float bc2 = (float)(1.0 - std::pow(beta2, step));
    fused_adamw_launcher(updated.data_ptr<float>(), new_m.data_ptr<float>(),
                         new_v.data_ptr<float>(), p.data_ptr<float>(), g.data_ptr<float>(),
                         m.data_ptr<float>(), v.data_ptr<float>(), (float)lr, (float)beta1,
                         (float)beta2, (float)eps, (float)weight_decay, bc1, bc2,
                         (int)p.numel(), c10::cuda::getCurrentCUDAStream().stream());
    return pybind11::make_tuple(updated, new_m, new_v);
}

static void register_fused_adamw(pybind11::module& m) {
    m.def("fused_adamw_forward", &fused_adamw_forward, py::arg("parameter"), py::arg("gradient"),
          py::arg("first_moment"), py::arg("second_moment"), py::arg("lr"), py::arg("beta1"),
          py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("step"));
}

REGISTER_BINDING(fused_adamw, register_fused_adamw);
