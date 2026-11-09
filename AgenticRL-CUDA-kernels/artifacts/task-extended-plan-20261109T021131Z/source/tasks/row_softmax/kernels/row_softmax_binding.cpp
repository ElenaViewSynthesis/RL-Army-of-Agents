#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void row_softmax_launcher(
    void* out, const void* x, long b, int h, int dtype_code, cudaStream_t stream
);

static torch::Tensor row_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "requires a 2D input");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
                x.scalar_type() == torch::kBFloat16, "requires float16, bfloat16 or float32 input");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    int code = x.scalar_type() == torch::kHalf ? 2 : x.scalar_type() == torch::kBFloat16 ? 1 : 0;
    row_softmax_launcher(out.data_ptr(), x.data_ptr(), (long)x.size(0), (int)x.size(1),
                         code, c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_row_softmax(pybind11::module& m) {
    m.def("row_softmax_forward", &row_softmax_forward, py::arg("x"));
}

REGISTER_BINDING(row_softmax, register_row_softmax);
