#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void grpo_group_norm_launcher(
    float* out, const float* x, int b, int g, float eps, cudaStream_t stream
);

static torch::Tensor grpo_group_norm_forward(torch::Tensor rewards, double eps) {
    TORCH_CHECK(rewards.is_cuda(), "rewards must be a CUDA tensor");
    TORCH_CHECK(rewards.is_contiguous(), "rewards must be contiguous");
    TORCH_CHECK(rewards.dim() == 2 && rewards.scalar_type() == torch::kFloat,
                "requires float32 [B,G] rewards");
    const c10::cuda::CUDAGuard guard(rewards.device());
    auto out = torch::empty_like(rewards);
    grpo_group_norm_launcher(out.data_ptr<float>(), rewards.data_ptr<float>(),
                             (int)rewards.size(0), (int)rewards.size(1), (float)eps,
                             c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_grpo_group_norm(pybind11::module& m) {
    m.def("grpo_group_norm_forward", &grpo_group_norm_forward,
          py::arg("rewards"), py::arg("eps"));
}

REGISTER_BINDING(grpo_group_norm, register_grpo_group_norm);
