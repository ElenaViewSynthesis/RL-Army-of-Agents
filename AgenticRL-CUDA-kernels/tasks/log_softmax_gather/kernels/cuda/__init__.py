"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["log_softmax_gather_forward"]


def __getattr__(name):
    if name == "log_softmax_gather_forward":
        from .log_softmax_gather import log_softmax_gather_forward
        return log_softmax_gather_forward
    raise AttributeError(name)
