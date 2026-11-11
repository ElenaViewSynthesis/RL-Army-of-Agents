"""Trusted Stage 6 lora_segment_metadata contract; see MULTI_LORA_STAGE6.md."""
import torch
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.lora_reference import forward


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_float32_matmul_precision("highest")

    def forward(self, *args):
        return forward('lora_segment_metadata', *args)


def get_inputs():
    return make_inputs('lora_segment_metadata')


def get_init_inputs():
    return []
