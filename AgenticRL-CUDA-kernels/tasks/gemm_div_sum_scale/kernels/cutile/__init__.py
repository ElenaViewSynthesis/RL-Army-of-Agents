"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["gemm_div_sum_scale_forward"]


def __getattr__(name):
    if name == "gemm_div_sum_scale_forward":
        from .gemm_div_sum_scale import gemm_div_sum_scale_forward
        return gemm_div_sum_scale_forward
    raise AttributeError(name)
