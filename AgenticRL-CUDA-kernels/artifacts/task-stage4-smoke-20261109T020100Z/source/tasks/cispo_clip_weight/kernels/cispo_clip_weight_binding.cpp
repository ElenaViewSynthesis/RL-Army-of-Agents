#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void cispo_clip_weight_launcher(
    float* out, const float* lr, const float* adv, const bool* mask,
    float low, float high, int n, int s, cudaStream_t stream
);

static torch::Tensor cispo_clip_weight_forward(torch::Tensor log_ratio, torch::Tensor advantage,
                                                 torch::Tensor mask, double low, double high) {
    TORCH_CHECK(log_ratio.is_cuda() && advantage.is_cuda() && mask.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(log_ratio.is_contiguous() && advantage.is_contiguous() && mask.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(log_ratio.dim() == 2 && advantage.dim() == 1 && mask.sizes() == log_ratio.sizes(),
                "requires log_ratio [B,S], advantage [B], mask [B,S]");
    TORCH_CHECK(advantage.size(0) == log_ratio.size(0), "advantage must have one entry per row");
    TORCH_CHECK(log_ratio.scalar_type() == torch::kFloat && advantage.scalar_type() == torch::kFloat,
                "log_ratio and advantage must be float32");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be boolean");
    TORCH_CHECK(log_ratio.device() == advantage.device() && log_ratio.device() == mask.device(),
                "inputs must be on the same device");
    const c10::cuda::CUDAGuard guard(log_ratio.device());
    auto out = torch::empty_like(log_ratio);
    cispo_clip_weight_launcher(out.data_ptr<float>(), log_ratio.data_ptr<float>(),
                               advantage.data_ptr<float>(), mask.data_ptr<bool>(),
                               (float)low, (float)high, (int)log_ratio.numel(),
                               (int)log_ratio.size(1), c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_cispo_clip_weight(pybind11::module& m) {
    m.def("cispo_clip_weight_forward", &cispo_clip_weight_forward,
          py::arg("log_ratio"), py::arg("advantage"), py::arg("mask"),
          py::arg("low"), py::arg("high"));
}

REGISTER_BINDING(cispo_clip_weight, register_cispo_clip_weight);
