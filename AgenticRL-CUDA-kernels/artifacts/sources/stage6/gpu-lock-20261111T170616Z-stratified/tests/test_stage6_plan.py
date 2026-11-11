"""Deterministic covering selection and physical memory sizing, without Torch."""

import json
import unittest

from scripts.plan_stage6 import planned_rows, size_case
from utils.lora_contracts import LORA_TASKS, slot_ranks, token_slots
from utils.task_catalog import get_task, select_cases


class Stage6PlanTests(unittest.TestCase):
    def test_stratified_covers_binding_axes_and_is_stable(self):
        for name in LORA_TASKS:
            selected = select_cases(name, 'stratified')
            self.assertEqual(len(selected), 6)
            self.assertEqual(len({c.id for c in selected}), 6)
            self.assertEqual({c.dtype for c in selected}, {'float16', 'bfloat16'})
            self.assertTrue({'none', 'mixed', 'single', 'distinct'} <= {c.params['pattern'] for c in selected})
            self.assertTrue({8, 256} <= {c.params['rank'] for c in selected})
            self.assertTrue({1, 3} <= {c.params['slices'] for c in selected})
            self.assertTrue(any(len(set(slot_ranks(c.params))) > 1 for c in selected))
            # Calls of other selections/tasks cannot affect the choice or order.
            select_cases(name, 'acceptance'); select_cases(LORA_TASKS[-1], 'all')
            self.assertEqual(selected, select_cases(name, 'stratified'))
            self.assertTrue(all(c in get_task(name).cases for c in selected))
            self.assertTrue(any(c.dtype == 'bfloat16' and c.params['rank'] == 256 for c in selected))
            none = next(c for c in selected if c.params['pattern'] == 'none')
            mixed = next(c for c in selected if c.params['pattern'] == 'mixed')
            self.assertEqual(token_slots(none.params), [-1]*none.params['tokens'])
            self.assertIn(-1, token_slots(mixed.params))
            self.assertTrue(any(i >= 0 for i in token_slots(mixed.params)))

    def test_decode_and_largest_case_skips_and_headroom(self):
        for name in LORA_TASKS:
            item = select_cases(name, 'acceptance')[0]
            self.assertIn(-1, token_slots(item.params))
            self.assertGreater(size_case(name, item)['gpu_headroom_bytes'], 32_000_000_000)
            if name.endswith('bgmv'):
                self.assertEqual(max(c.params['tokens'] for c in get_task(name).cases), 256)
                self.assertEqual(item.params['tokens'], 64)
            else:
                self.assertEqual(item.params['tokens'], 4096)
            # Input and output counts must agree with the actual padded ABI.
            sized = size_case(name, item)
            if 'expand' in name or 'fused' in name:
                p = item.params
                self.assertEqual(sized['output_bytes'], 2*p['tokens']*p['slices']*(p['out']+16))
            json.dumps(sized, allow_nan=False)

    def test_all_fixtures_are_enumerated_and_applicability_is_honest(self):
        rows = planned_rows('acceptance')
        self.assertEqual(len(rows), 56)
        self.assertTrue(all(r['applicable'] for r in rows))
        rows = planned_rows('stratified')
        self.assertEqual(len(rows), 336)
        self.assertEqual(sum(r['applicable'] for r in rows), 312)
        for row in rows:
            if not row['applicable']:
                self.assertIn(row['fixture'], ('wrong_skip', 'wrong_padding'))
        self.assertFalse(any(r['fixture'] == 'fp16_intermediate' for r in rows))
