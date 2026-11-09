#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void packed_varlen_reduction_launcher(
    float* out, const float* v, const int* off, int b, int is_mean, cudaStream_t stream
);

static torch::Tensor packed_varlen_reduction_forward(torch::Tensor values, torch::Tensor offsets,
                                                       bool is_mean) {
    TORCH_CHECK(values.is_cuda() && offsets.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(values.is_contiguous() && offsets.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(values.dim() == 1 && offsets.dim() == 1 && offsets.numel() >= 1,
                "requires values [N] and offsets [B+1]");
    TORCH_CHECK(values.device() == offsets.device(), "inputs must be on the same device");
    TORCH_CHECK(values.scalar_type() == torch::kFloat, "values must be float32");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt, "offsets must be int32");
    const c10::cuda::CUDAGuard guard(values.device());
    int64_t b = offsets.numel() - 1;
    auto out = torch::empty({b}, values.options());
    packed_varlen_reduction_launcher(out.data_ptr<float>(), values.data_ptr<float>(),
                                     offsets.data_ptr<int>(), (int)b, is_mean ? 1 : 0,
                                     c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_packed_varlen_reduction(pybind11::module& m) {
    m.def("packed_varlen_reduction_forward", &packed_varlen_reduction_forward,
          py::arg("values"), py::arg("offsets"), py::arg("is_mean"));
}

REGISTER_BINDING(packed_varlen_reduction, register_packed_varlen_reduction);
