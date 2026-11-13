#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

extern "C" void relu_squared_launcher(void* out, const void* x, long rows, long width, int input_is_bf16,
                                cudaStream_t stream);

static torch::Tensor relu_squared_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous() && x.dim() == 2, "requires a contiguous 2-D input");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "requires float32 or bfloat16 input");
    
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty({x.size(0), x.size(1)}, x.options());
    relu_squared_launcher(out.data_ptr(), x.data_ptr(), x.size(0), x.size(1),
                    x.scalar_type() == torch::kBFloat16 ? 1 : 0,
                    c10::cuda::getCurrentCUDAStream().stream());
    return out;
}

static void register_relu_squared(pybind11::module& m) {
    m.def("relu_squared_forward", &relu_squared_forward, py::arg("x"));
}

REGISTER_BINDING(relu_squared, register_relu_squared);
