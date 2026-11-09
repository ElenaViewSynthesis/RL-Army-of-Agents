"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["grouped_gemm_forward"]


def __getattr__(name):
    if name == "grouped_gemm_forward":
        from .grouped_gemm import grouped_gemm_forward
        return grouped_gemm_forward
    raise AttributeError(name)
