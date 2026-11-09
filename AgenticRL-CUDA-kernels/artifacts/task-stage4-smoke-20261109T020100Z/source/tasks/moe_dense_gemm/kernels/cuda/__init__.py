"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["moe_dense_gemm_forward"]


def __getattr__(name):
    if name == "moe_dense_gemm_forward":
        from .moe_dense_gemm import moe_dense_gemm_forward
        return moe_dense_gemm_forward
    raise AttributeError(name)
