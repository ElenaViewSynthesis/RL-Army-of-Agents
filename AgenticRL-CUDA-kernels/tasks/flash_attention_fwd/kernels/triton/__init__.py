"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["flash_attention_forward"]


def __getattr__(name):
    if name == "flash_attention_forward":
        from .flash_attention_fwd import flash_attention_forward
        return flash_attention_forward
    raise AttributeError(name)
