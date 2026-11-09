import torch


def packed_varlen_reduction_forward(values, offsets, reduction):
    import cuda_extension
    if values.ndim != 1 or offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("packed_varlen_reduction requires values [N] and offsets [B+1]")
    if values.dtype != torch.float32:
        raise ValueError("values must be float32")
    if offsets.dtype != torch.int32:
        raise ValueError("offsets must be int32")
    if values.device != offsets.device:
        raise ValueError("values and offsets must share a device")
    if not values.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("packed_varlen_reduction requires contiguous inputs")
    return cuda_extension.packed_varlen_reduction_forward(values, offsets,
                                                          reduction == "mean")
