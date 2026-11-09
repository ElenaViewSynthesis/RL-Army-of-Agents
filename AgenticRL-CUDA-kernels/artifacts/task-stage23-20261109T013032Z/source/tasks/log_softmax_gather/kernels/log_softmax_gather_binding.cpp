#include <cstdint>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void log_softmax_gather_launcher(
    float* out, float* pmax, float* psum, const void* x, const int64_t* ids,
    int b, int v, int input_is_bf16, cudaStream_t stream
);

static torch::Tensor log_softmax_gather_forward(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.is_cuda() && token_ids.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(logits.is_contiguous() && token_ids.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(logits.dim() == 2 && token_ids.dim() == 1 && token_ids.size(0) == logits.size(0),
                "requires logits [B,V] and token_ids [B]");
    TORCH_CHECK(logits.device() == token_ids.device(), "inputs must be on the same device");
    TORCH_CHECK(token_ids.scalar_type() == torch::kLong, "token_ids must be int64");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat || logits.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 input");
    const c10::cuda::CUDAGuard guard(logits.device());
    int64_t b = logits.size(0), v = logits.size(1);
    int64_t chunks = (v + 4095) / 4096;
    auto fp = logits.options().dtype(torch::kFloat);
    auto out = torch::empty({b}, fp);
    auto pmax = torch::empty({b, chunks}, fp);
    auto psum = torch::empty({b, chunks}, fp);
    log_softmax_gather_launcher(out.data_ptr<float>(), pmax.data_ptr<float>(),
                                psum.data_ptr<float>(), logits.data_ptr(),
                                token_ids.data_ptr<int64_t>(),
                                (int)b, (int)v,
                                logits.scalar_type() == torch::kBFloat16 ? 1 : 0,
                                c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_log_softmax_gather(pybind11::module& m) {
    m.def("log_softmax_gather_forward", &log_softmax_gather_forward,
          py::arg("logits"), py::arg("token_ids"));
}

REGISTER_BINDING(log_softmax_gather, register_log_softmax_gather);
