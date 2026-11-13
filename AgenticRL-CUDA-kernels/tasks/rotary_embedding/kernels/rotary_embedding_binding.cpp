#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void rotary_embedding_launcher(void* out, const void* x, const void* cos_table,
                                          const void* sin_table, long n_rows, long seq, long half,
                                          int input_is_bf16, cudaStream_t stream);

static torch::Tensor rotate_one(const torch::Tensor& x, const torch::Tensor& cos_table,
                                const torch::Tensor& sin_table) {
    auto out = torch::empty_like(x);
    long seq = x.size(2), half = x.size(3) / 2;
    rotary_embedding_launcher(out.data_ptr(), x.data_ptr(), cos_table.data_ptr(), sin_table.data_ptr(),
                              x.numel() / (2 * half), seq, half,
                              x.scalar_type() == torch::kBFloat16 ? 1 : 0,
                              c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static std::tuple<torch::Tensor, torch::Tensor> rotary_embedding_forward(
        torch::Tensor q, torch::Tensor k, torch::Tensor cos_table, torch::Tensor sin_table) {
    for (const auto& t : {q, k, cos_table, sin_table}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "inputs must be contiguous CUDA tensors");
    }
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.size(3) % 2 == 0, "requires q,k [B,H,S,D] with even D");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share a dtype");
    TORCH_CHECK(q.scalar_type() == torch::kFloat || q.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 q and k");
    TORCH_CHECK(cos_table.scalar_type() == torch::kFloat && sin_table.scalar_type() == torch::kFloat,
                "requires float32 cos/sin tables");
    const c10::cuda::CUDAGuard guard(q.device());
    return {rotate_one(q, cos_table, sin_table), rotate_one(k, cos_table, sin_table)};
}

static void register_rotary_embedding(pybind11::module& m) {
    m.def("rotary_embedding_forward", &rotary_embedding_forward,
          py::arg("q"), py::arg("k"), py::arg("cos_table"), py::arg("sin_table"));
}

REGISTER_BINDING(rotary_embedding, register_rotary_embedding);
