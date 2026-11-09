#include <algorithm>
#include <cstdint>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"
extern "C" void topk_sampling_launcher(int*, uint64_t*, const float*, const float*, int, int, int, int, float, cudaStream_t);
static torch::Tensor topk_forward(torch::Tensor logits, torch::Tensor uniforms, int64_t top_k, double top_p) {
    TORCH_CHECK(logits.is_cuda() && uniforms.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(logits.is_contiguous() && uniforms.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(logits.dim() == 2 && uniforms.dim() == 1 && uniforms.size(0) == logits.size(0), "requires logits [B,V], uniforms [B]");
    TORCH_CHECK(logits.device() == uniforms.device(), "inputs must share a device");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat && uniforms.scalar_type() == torch::kFloat, "requires float32 inputs");
    TORCH_CHECK(logits.size(1) > 0 && top_k >= 1 && top_k <= std::min<int64_t>(32, logits.size(1)), "top_k must be in [1,min(32,V)] and V>0");
    TORCH_CHECK(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0,1]");
    const c10::cuda::CUDAGuard guard(logits.device());
    int64_t b = logits.size(0), v = logits.size(1);
    int64_t chunks = (v + 1023) / 1024;
    auto out = torch::empty({b}, logits.options().dtype(torch::kInt));
    auto scratch = torch::empty({b * chunks * top_k}, logits.options().dtype(torch::kLong));
    topk_sampling_launcher(out.data_ptr<int>(), (uint64_t*)scratch.data_ptr<int64_t>(),
                           logits.data_ptr<float>(), uniforms.data_ptr<float>(),
                           (int)b, (int)v, (int)top_k, (int)chunks, (float)top_p,
                           c10::cuda::getCurrentCUDAStream().stream());
    return out;
}
static void reg(pybind11::module& m) { m.def("topk_sampling_forward", &topk_forward, py::arg("logits"), py::arg("uniforms"), py::arg("top_k"), py::arg("top_p")); }
REGISTER_BINDING(topk_sampling, reg);
