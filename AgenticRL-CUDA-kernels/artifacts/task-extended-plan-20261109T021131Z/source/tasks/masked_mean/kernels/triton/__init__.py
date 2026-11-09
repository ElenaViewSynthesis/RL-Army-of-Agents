"""Triton kernel entry points; imported only inside kernel workers."""

__all__ = ["masked_mean_forward"]


def __getattr__(name):
    if name == "masked_mean_forward":
        from .masked_mean import masked_mean_forward
        return masked_mean_forward
    raise AttributeError(name)
