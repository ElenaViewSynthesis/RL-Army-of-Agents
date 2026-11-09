#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void prefix_sum_launcher(
    void* out, const void* x, int n, int input_is_int32, cudaStream_t stream
);

static torch::Tensor prefix_sum_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 1, "requires a 1D input");
    TORCH_CHECK(x.scalar_type() == torch::kInt || x.scalar_type() == torch::kFloat,
                "requires int32 or float32 input");
    TORCH_CHECK(x.numel() <= 0x7fffffff, "input too large");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    prefix_sum_launcher(out.data_ptr(), x.data_ptr(), (int)x.numel(),
                        x.scalar_type() == torch::kInt ? 1 : 0,
                        c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_prefix_sum(pybind11::module& m) {
    m.def("prefix_sum_forward", &prefix_sum_forward, py::arg("x"));
}

REGISTER_BINDING(prefix_sum, register_prefix_sum);
