from model import Model
from utils.lora_controls import control

BACKEND = "triton"


class ModelNew(Model):
    def forward(self, *args):
        return control('lora_expand_bgmv', 'wrong_skip', *args)
