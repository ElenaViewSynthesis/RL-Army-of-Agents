"""Approved integer-only strengthening, without changing stock float tolerances."""

import importlib.util
import unittest

from tracking.contract import classify
from utils.config import RolloutConfig


class IntegerConfigTests(unittest.TestCase):
    def test_default_stock_config_unchanged(self):
        self.assertFalse(RolloutConfig().exact_integer_outputs)
        self.assertEqual(classify("cfg", "cfg/exact_integer_outputs"), "config")
        with self.assertRaises(ValueError):
            RolloutConfig(exact_integer_outputs="false")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
class IntegerComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from utils import task_verification, verification

        cls.torch, cls.task, cls.stock = torch, task_verification, verification

    def test_stock_comparison_is_not_silently_changed(self):
        self.stock.check_equal(self.torch.tensor([100001]), self.torch.tensor([100000]))

    def test_token_and_prefix_off_by_one_rejected(self):
        for dtype in (self.torch.int32, self.torch.int64):
            for value in (100000, 10000000):
                with self.subTest(dtype=dtype, value=value):
                    with self.assertRaises(AssertionError):
                        self.task.check_equal(self.torch.tensor([value + 1], dtype=dtype),
                                              self.torch.tensor([value], dtype=dtype))

    def test_equal_integer_arrays_pass(self):
        expected = self.torch.tensor([0, 1, 10000000], dtype=self.torch.int32)
        self.task.check_equal(expected.clone(), expected)

    def test_float_tolerance_retains_one_percent_relative_and_absolute(self):
        target = self.torch.tensor([1.])
        self.task.check_equal(self.torch.tensor([1.019]), target)
        with self.assertRaises(AssertionError):
            self.task.check_equal(self.torch.tensor([1.021]), target)

    def test_nested_integer_leaf_cannot_hide_in_float_container(self):
        target = {"result": (self.torch.tensor([100000], dtype=self.torch.int32),
                             [self.torch.tensor([1.])])}
        actual = {"result": (self.torch.tensor([100001], dtype=self.torch.int32),
                             [self.torch.tensor([1.019])])}
        with self.assertRaises(AssertionError):
            self.task.check_equal(actual, target)

    def test_structure_and_dtype_checks_are_preserved(self):
        with self.assertRaises(AssertionError):
            self.task.check_equal([self.torch.tensor([1])], (self.torch.tensor([1]),))
        with self.assertRaises(AssertionError):
            self.task.check_equal(self.torch.tensor([1.]), self.torch.tensor([1], dtype=self.torch.int32))


if __name__ == "__main__":
    unittest.main()
