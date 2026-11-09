"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["fused_adamw_forward"]


def __getattr__(name):
    if name == "fused_adamw_forward":
        from .fused_adamw import fused_adamw_forward
        return fused_adamw_forward
    raise AttributeError(name)
