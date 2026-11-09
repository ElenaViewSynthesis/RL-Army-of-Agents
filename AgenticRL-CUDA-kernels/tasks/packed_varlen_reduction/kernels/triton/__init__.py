"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["packed_varlen_reduction_forward"]


def __getattr__(name):
    if name == "packed_varlen_reduction_forward":
        from .packed_varlen_reduction import packed_varlen_reduction_forward
        return packed_varlen_reduction_forward
    raise AttributeError(name)
