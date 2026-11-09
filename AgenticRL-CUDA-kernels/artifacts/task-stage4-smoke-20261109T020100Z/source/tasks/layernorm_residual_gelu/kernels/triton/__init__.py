"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["layernorm_residual_gelu_forward"]


def __getattr__(name):
    if name == "layernorm_residual_gelu_forward":
        from .layernorm_residual_gelu import layernorm_residual_gelu_forward
        return layernorm_residual_gelu_forward
    raise AttributeError(name)
