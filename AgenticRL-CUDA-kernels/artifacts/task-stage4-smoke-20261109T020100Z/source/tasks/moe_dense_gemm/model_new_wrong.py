from model import Model
from utils.task_fixtures import wrong_output

BACKEND = "triton"


class ModelNew(Model):
    def forward(self, *args):
        return wrong_output(super().forward(*args))
