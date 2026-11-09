#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void masked_mean_launcher(
    float* out, const void* x, const int* lengths, int b, int l, int input_is_bf16,
    cudaStream_t stream
);

static torch::Tensor masked_mean_forward(torch::Tensor x, torch::Tensor lengths) {
    TORCH_CHECK(x.is_cuda() && lengths.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && lengths.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && lengths.dim() == 1 && lengths.size(0) == x.size(0),
                "requires x [B,L] and lengths [B]");
    TORCH_CHECK(lengths.scalar_type() == torch::kInt, "lengths must be int32");
    TORCH_CHECK(x.device() == lengths.device(), "x and lengths must be on the same device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 input");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat));
    masked_mean_launcher(out.data_ptr<float>(), x.data_ptr(), lengths.data_ptr<int>(),
                         (int)x.size(0), (int)x.size(1),
                         x.scalar_type() == torch::kBFloat16 ? 1 : 0,
                         c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_masked_mean(pybind11::module& m) {
    m.def("masked_mean_forward", &masked_mean_forward, py::arg("x"), py::arg("lengths"));
}

REGISTER_BINDING(masked_mean, register_masked_mean);
