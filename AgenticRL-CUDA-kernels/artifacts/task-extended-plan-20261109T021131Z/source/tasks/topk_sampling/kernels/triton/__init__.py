"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["topk_sampling_forward"]


def __getattr__(name):
    if name == "topk_sampling_forward":
        from .topk_sampling import topk_sampling_forward
        return topk_sampling_forward
    raise AttributeError(name)
