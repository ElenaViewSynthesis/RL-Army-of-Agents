#include <cmath>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"
extern "C" void decode_attention_launcher(void*, float*, const void*, const void*, const void*, int, int, int, int, int, float, cudaStream_t);
static torch::Tensor da_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "requires q [B,Hq,1,D], k,v [B,Hkv,S,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3) && q.size(2) == 1, "shape mismatch");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0, "query heads must divide evenly over kv heads");
    TORCH_CHECK(q.size(3) == 64, "requires head dim D=64");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "inputs must share a device");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "requires bfloat16 inputs");
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q);
    int64_t b = q.size(0), hq = q.size(1), hkv = k.size(1), s = k.size(2), d = q.size(3);
    auto scores = torch::empty({b * hq, s}, q.options().dtype(torch::kFloat));
    decode_attention_launcher(out.data_ptr(), scores.data_ptr<float>(), q.data_ptr(), k.data_ptr(),
                              v.data_ptr(), (int)b, (int)hq, (int)hkv, (int)s, (int)d,
                              (float)(1.0 / std::sqrt((double)d)), c10::cuda::getCurrentCUDAStream().stream());
    return out;
}
static void reg(pybind11::module& m) { m.def("decode_attention_forward", &da_forward, py::arg("q"), py::arg("k"), py::arg("v")); }
REGISTER_BINDING(decode_attention, reg);
