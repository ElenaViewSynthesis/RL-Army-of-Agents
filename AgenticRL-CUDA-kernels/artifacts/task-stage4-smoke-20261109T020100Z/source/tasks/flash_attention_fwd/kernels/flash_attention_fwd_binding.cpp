#include <cmath>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"
extern "C" void fa_launcher(void*, float*, const void*, const void*, const void*, int, int, int, float, int, cudaStream_t);
static torch::Tensor fa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.sizes() == q.sizes(), "requires q,k,v [B,H,S,D]");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "inputs must share a device");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "requires bfloat16 inputs");
    TORCH_CHECK(q.size(3) == 64, "requires head dim D=64");
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q);
    int64_t s = q.size(2), d = q.size(3), bh = q.size(0) * q.size(1);
    auto scores = torch::empty({bh, s, s}, q.options().dtype(torch::kFloat));
    fa_launcher(out.data_ptr(), scores.data_ptr<float>(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
                (int)bh, (int)s, (int)d, (float)(1.0 / std::sqrt((double)d)), causal ? 1 : 0,
                c10::cuda::getCurrentCUDAStream().stream());
    return out;
}
static void reg(pybind11::module& m) { m.def("flash_attention_forward", &fa_forward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal")); }
REGISTER_BINDING(flash_attention_fwd, reg);
