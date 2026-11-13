"""Intentional Torch-op negative controls; never imported by honest kernels."""


def wrong_output(output):
    import torch

    if isinstance(output, torch.Tensor):
        return torch.full_like(output, float("nan") if output.is_floating_point() else -2147483648)
    if isinstance(output, tuple):
        return tuple(wrong_output(value) for value in output)
    if isinstance(output, list):
        return [wrong_output(value) for value in output]
    if isinstance(output, dict):
        return {key: wrong_output(value) for key, value in output.items()}
    raise TypeError(f"Unsupported negative-control output {type(output)}")


def negative_sources(task_name):
    from utils.task_catalog import get_task

    get_task(task_name)
    hack = '''from model import Model

BACKEND = "triton"


class ModelNew(Model):
    pass
'''
    if task_name == "flash_attention_fwd":
        hack = '''import torch.nn.functional as F
from model import Model

BACKEND = "triton"
_sdpa = F.scaled_dot_product_attention


class ModelNew(Model):
    def forward(self, q, k, v):
        return _sdpa(q, k, v, dropout_p=0.0, is_causal=self.causal)
'''
    wrong = '''from model import Model
from utils.task_fixtures import wrong_output

BACKEND = "triton"


class ModelNew(Model):
    def forward(self, *args):
        return wrong_output(super().forward(*args))
'''
    return {"model_new_hack.py": hack, "model_new_wrong.py": wrong}


def main():
    import argparse
    from pathlib import Path

    from utils.task_catalog import TASKS

    parser = argparse.ArgumentParser(description="Create missing deliberate negative controls; never overwrite")
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / "tasks"
    for name in TASKS if args.tasks == "all" else args.tasks.split(","):
        for filename, source in negative_sources(name).items():
            path = root / name / filename
            if not path.exists():
                with path.open("x") as stream:
                    stream.write(source)


if __name__ == "__main__":
    main()
