"""CUDA extension entry points; imported only inside kernel workers."""

__all__ = ["prefix_sum_forward"]


def __getattr__(name):
    if name == "prefix_sum_forward":
        from .prefix_sum import prefix_sum_forward
        return prefix_sum_forward
    raise AttributeError(name)
