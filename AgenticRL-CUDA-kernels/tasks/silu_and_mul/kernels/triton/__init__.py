"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["silu_and_mul_forward"]


def __getattr__(name):
    if name == "silu_and_mul_forward":
        from .silu_and_mul import silu_and_mul_forward
        return silu_and_mul_forward
    raise AttributeError(name)
