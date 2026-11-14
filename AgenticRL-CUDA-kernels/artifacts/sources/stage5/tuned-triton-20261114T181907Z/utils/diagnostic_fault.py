import triton
import triton.language as tl


@triton.jit
def diagnostic_invalid_store(pointer):
    tl.store(pointer + 4096, 1.0)


def run():
    import torch
    from utils.backend import DEVICE

    pointer = torch.empty(1, dtype=torch.float32, device=DEVICE)
    diagnostic_invalid_store[(1,)](pointer)
    torch.cuda.synchronize()
