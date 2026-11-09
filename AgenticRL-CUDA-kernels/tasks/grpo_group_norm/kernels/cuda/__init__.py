"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["grpo_group_norm_forward"]


def __getattr__(name):
    if name == "grpo_group_norm_forward":
        from .grpo_group_norm import grpo_group_norm_forward
        return grpo_group_norm_forward
    raise AttributeError(name)
