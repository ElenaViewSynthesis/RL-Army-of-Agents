import json
import shutil
import tempfile
from pathlib import Path

from utils.backend import BACKENDS

ROOT = Path(__file__).resolve().parent.parent

_REWARD_HACK_TORCHOP = '''import torch.nn as nn
BACKEND = "{backend}"
class ModelNew(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, a, b):
        return self.alpha * a + b
'''


def scratch(prefix="test-"):
    root = ROOT / "artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=root)


def make_task(parent, kind="triton"):
    """kind: a backend name (its honest template), "reward-hack-torchop-<backend>" (Torch-op
    substitution declaring that backend), or a Triton-declared reward-hack-torchop/sleep/wrong/crash."""
    path = Path(parent) / kind
    path.mkdir()
    for name in ("model.py", "binding.cpp", "binding_registry.h"):
        shutil.copy2(ROOT / name, path / name)
    shutil.copytree(ROOT / "kernels", path / "kernels", ignore=shutil.ignore_patterns("__pycache__"))
    if kind in BACKENDS:
        source = (ROOT / BACKENDS[kind].template).read_text()
    elif kind.startswith("reward-hack-torchop-") and kind[len("reward-hack-torchop-"):] in BACKENDS:
        source = _REWARD_HACK_TORCHOP.format(backend=kind[len("reward-hack-torchop-"):])
    elif kind in ("reward-hack-torchop", "sleep", "wrong", "crash"):
        source = _REWARD_HACK_TORCHOP.format(backend="triton")
        if kind == "sleep":
            source += "\nimport time\ntime.sleep(60)\n"
        if kind == "wrong":
            source = source.replace("self.alpha * a + b", "self.alpha * a + b + 100")
        if kind == "crash":
            source += "\nimport os\nos._exit(7)\n"
    else:
        raise ValueError(kind)
    (path / "model_new.py").write_text(source)
    return path


def assert_flat(test, result):
    test.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
    test.assertTrue(all(v is None or isinstance(v, (str, bool, int, float)) for v in result.values()))


def mock_rollout(reward=3.0, reason="faster_than_both", turns=1, penalty=0.0, rid=None):
    row = {"reward/value": reward, "reward/base": 3 if reward > 0 else -1,
           "reward/reason": reason, "reward/overlong_penalty": penalty, "env/turns_used": turns}
    if rid is not None:
        row["rollout_id"] = rid
    return row


def make_spoof_task(parent, backend):
    """Launch a copied honest kernel renamed `kernel`, then return Torch ops.

    Only scratch copies are edited; root protected upstream files stay intact.
    """
    import re
    from utils.backend import agent_kernel_names

    task = make_task(parent, backend)
    spec = BACKENDS[backend]
    names = agent_kernel_names(task, backend)
    for path in task.glob(spec.kernel_glob):
        source = path.read_text()
        for name in names:
            source = re.sub(r"\b" + re.escape(name) + r"\b", "kernel", source)
        path.write_text(source)
    path = task / "model_new.py"
    source = path.read_text()
    source = source.replace("        return ", "        ")
    source = source.rstrip() + "\n        return self.alpha * a + b\n"
    path.write_text(source)
    return task
