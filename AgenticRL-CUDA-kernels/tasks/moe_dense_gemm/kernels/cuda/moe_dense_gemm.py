import torch
def moe_dense_gemm_forward(x, gate, up, down, alpha):
    import cuda_extension
    if x.ndim != 2 or gate.ndim != 3 or up.shape != gate.shape or down.ndim != 3:
        raise ValueError("moe_dense_gemm requires x [T,H], gate/up [E,H,I], down [E,I,H]")
    e, h, i = gate.shape
    if x.shape[1] != h or down.shape != (e, i, h) or alpha.shape != (x.shape[0], e):
        raise ValueError("moe_dense_gemm shape mismatch")
    if not all(t.dtype == torch.bfloat16 for t in (x, gate, up, down)) or alpha.dtype != torch.float32:
        raise ValueError("moe_dense_gemm requires bfloat16 weights and float32 alpha")
    if not all(t.device == x.device for t in (gate, up, down, alpha)):
        raise ValueError("inputs must share a device")
    if not all(t.is_contiguous() for t in (x, gate, up, down, alpha)):
        raise ValueError("moe_dense_gemm requires contiguous inputs")
    return cuda_extension.moe_dense_gemm_forward(x, gate, up, down, alpha)
