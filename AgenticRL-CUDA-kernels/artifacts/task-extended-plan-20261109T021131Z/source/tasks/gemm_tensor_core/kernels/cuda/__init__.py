"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["gemm_tensor_core_forward"]


def __getattr__(name):
    if name == "gemm_tensor_core_forward":
        from .gemm_tensor_core import gemm_tensor_core_forward
        return gemm_tensor_core_forward
    raise AttributeError(name)
