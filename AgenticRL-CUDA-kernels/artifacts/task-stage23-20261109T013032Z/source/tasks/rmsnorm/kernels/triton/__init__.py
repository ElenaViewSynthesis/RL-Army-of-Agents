"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["rmsnorm_forward"]


def __getattr__(name):
    if name == "rmsnorm_forward":
        from .rmsnorm import rmsnorm_forward
        return rmsnorm_forward
    raise AttributeError(name)
