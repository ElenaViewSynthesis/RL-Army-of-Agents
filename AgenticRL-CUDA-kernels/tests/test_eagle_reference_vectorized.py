"""The vectorized Stage 5 references must match the original per-element loops exactly.

tests/fixtures/eagle_reference_loop.py is a frozen copy of the original
references, which walked trees with per-element host syncs (int(tensor)).
Those were correct but made the eager GPU baseline meaningless and overflowed
the profiler's 4 MiB stdout tail at the largest cases (H200 run
task-stage5-gpu-acceptance-20261109T190928Z). Inputs here are SYNTHETIC.
"""

import importlib.util
import itertools
import types
import unittest

from tests.support import ROOT

HAS_TORCH = importlib.util.find_spec("torch") is not None
SHAPES = ("chain", "wide", "balanced", "ragged")


def _oracle():
    spec = importlib.util.spec_from_file_location(
        "eagle_reference_loop", ROOT / "tests" / "fixtures" / "eagle_reference_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(**params):
    return types.SimpleNamespace(params=params)


@unittest.skipUnless(HAS_TORCH, "requires torch (CPU is enough)")
class VectorizedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from utils import eagle_reference
        cls.torch, cls.new, cls.old = torch, eagle_reference, _oracle()

    def assertSame(self, new, old):
        torch = self.torch
        new = new if isinstance(new, tuple) else (new,)
        old = old if isinstance(old, tuple) else (old,)
        self.assertEqual(len(new), len(old))
        for a, b in zip(new, old):
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
            self.assertTrue(torch.equal(a, b), f"max diff {(a.double() - b.double()).abs().max()}")

    def grid(self):
        return itertools.product((1, 3), (1, 2, 5, 17, 64), SHAPES)

    def inputs(self, name, b, n, shape, **extra):
        self.torch.manual_seed(b * 1000 + n)
        return self.new.make_inputs(name, _item(b=b, n=n, tree=shape, **extra))

    def test_ancestors_and_tree_step(self):
        for b, n, shape in self.grid():
            with self.subTest(b=b, n=n, shape=shape):
                x = self.inputs("eagle_tree_step", b, n, shape, top_k=4)
                self.assertSame(self.new.ancestors(x[2]), self.old.ancestors(x[2]))
                self.assertSame(self.new.tree_step(*x), self.old.tree_step(*x))

    def test_attention(self):
        for b, n, shape in itertools.product((1, 2), (1, 9, 64), SHAPES):
            with self.subTest(b=b, n=n, shape=shape):
                x = self.inputs("eagle_tree_row_attention", b, n, shape, hkv=2, ratio=2, d=16, s=24)
                self.assertSame(self.new.attention(*x, 8), self.old.attention(*x, 8))

    def test_tree_conv_gdn_and_mamba(self):
        for (b, n, shape), name in itertools.product(self.grid(), ("eagle_gdn_tree_conv",
                                                                   "eagle_mamba2_tree_conv")):
            with self.subTest(b=b, n=n, shape=shape, name=name):
                x = self.inputs(name, b, n, shape, c=8)
                mamba = name == "eagle_mamba2_tree_conv"
                self.assertSame(self.new.tree_conv(*x, mamba=mamba), self.old.tree_conv(*x, mamba=mamba))

    def test_recurrence(self):
        for b, n, shape in itertools.product((1, 3), (1, 5, 17, 64), SHAPES):
            with self.subTest(b=b, n=n, shape=shape):
                x = self.inputs("eagle_gdn_tree_recurrence", b, n, shape, heads=2, dk=4, dv=3)
                self.assertSame(self.new.recurrence(*x), self.old.recurrence(*x))

    def test_greedy_accept(self):
        for b, n, shape in self.grid():
            with self.subTest(b=b, n=n, shape=shape):
                x = self.inputs("eagle_accept_tail", b, n, shape, v=50)
                self.assertSame(self.new.accept(*x), self.old.accept(*x))

    def test_sampled_accept(self):
        for b, n, shape in itertools.product((1, 3), (1, 5, 17, 64), SHAPES):
            with self.subTest(b=b, n=n, shape=shape):
                x = self.inputs("eagle_accept_tail_sampled", b, n, shape, v=64, seed=17)
                self.assertSame(self.new.accept_sampled(*x), self.old.accept_sampled(*x))

    def test_no_host_sync_in_tree_references(self):
        # A host sync on a CUDA tensor would show up as int()/float()/.nonzero()/.item().
        import inspect
        import re
        for name in ("ancestors", "tree_conv", "recurrence", "_walk", "accept", "accept_sampled"):
            source = inspect.getsource(getattr(self.new, name))
            self.assertIsNone(re.search(r"(?<![.\w])(int|float|bool)\(|\.(nonzero|item|tolist)\(", source),
                              f"{name} has a host sync")


class TritonGridLimitTests(unittest.TestCase):
    def test_vocabulary_is_on_grid_axis_zero(self):
        # Grid axes 1 and 2 are capped at 65535; V reaches 151936 (H200 run
        # task-stage5-gpu-acceptance-20261109T190928Z failed with "invalid argument").
        source = (ROOT / "tasks/eagle_candidate_topk/kernels/triton/candidate_topk.py").read_text()
        self.assertIn("eagle_vocab_score[(v,b)](", source)
        self.assertIn("token, batch = tl.program_id(0), tl.program_id(1)", source)


if __name__ == "__main__":
    unittest.main()
