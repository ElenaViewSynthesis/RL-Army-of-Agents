import json
import shutil
import tempfile
from pathlib import Path

from utils.backend import BACKENDS

ROOT = Path(__file__).resolve().parent.parent

_HACK = '''import torch.nn as nn
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
    """kind: a backend name (its honest template), "hack-<backend>" (Torch-op
    substitution declaring that backend), or a Triton-declared hack/sleep/wrong/crash."""
    path = Path(parent) / kind
    path.mkdir()
    for name in ("model.py", "binding.cpp", "binding_registry.h"):
        shutil.copy2(ROOT / name, path / name)
    shutil.copytree(ROOT / "kernels", path / "kernels", ignore=shutil.ignore_patterns("__pycache__"))
    if kind in BACKENDS:
        source = (ROOT / BACKENDS[kind].template).read_text()
    elif kind.startswith("hack-") and kind[len("hack-"):] in BACKENDS:
        source = _HACK.format(backend=kind[len("hack-"):])
    elif kind in ("hack", "sleep", "wrong", "crash"):
        source = _HACK.format(backend="triton")
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
