"""Triton kernel entry points; imported only inside kernel workers."""

__all__ = ["row_softmax_forward"]


def __getattr__(name):
    if name == "row_softmax_forward":
        from .row_softmax import row_softmax_forward
        return row_softmax_forward
    raise AttributeError(name)
