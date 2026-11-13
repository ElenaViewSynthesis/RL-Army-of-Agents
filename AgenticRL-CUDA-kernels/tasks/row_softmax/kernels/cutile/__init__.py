"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["row_softmax_forward"]


def __getattr__(name):
    if name == "row_softmax_forward":
        from .row_softmax import row_softmax_forward
        return row_softmax_forward
    raise AttributeError(name)
