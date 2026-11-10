from model import Model
from utils.lora_controls import control

BACKEND = "triton"


class ModelNew(Model):
    def forward(self, *args):
        return control('lora_fused_sgmv', 'wrong_padding', *args)
