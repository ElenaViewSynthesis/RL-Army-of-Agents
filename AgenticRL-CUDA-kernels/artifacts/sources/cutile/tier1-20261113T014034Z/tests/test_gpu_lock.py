"""Synthetic device discovery and real OS-lock/process-death tests; no GPU."""

import argparse
import dataclasses
import fcntl
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import ROOT, mock_rollout
from utils.config import RolloutConfig
from utils.gpu_lock import MeasurementLock, device_identity
from utils.grpo import build_batch, group_advantages
from utils.rollout_env import run_task


HOLDER = '''
import os
from utils.gpu_lock import MeasurementLock
with MeasurementLock(os.environ, timeout_s=8) as lock:
    assert lock.acquired
    print("HELD", flush=True)
    os.read(0, 1)
'''

WAITER = '''
import json, os, sys
from unittest.mock import patch
from utils.config import RolloutConfig
from utils.rollout_env import run_task
deadlines = []
def stage(module, args, path, env, deadline):
    deadlines.append(deadline)
    if module == "utils.verification":
        print("VERIFIED", flush=True)
        return 0, "[PASS] verify success", "", False
    if module == "utils.profiling":
        return 0, json.dumps(dict(t_eager_us=100, t_compile_us=80, t_gen_us=50, provenance=1)), "", False
    return 0, "", "", False
with patch("utils.rollout_env._stage", side_effect=stage):
    result = run_task(sys.argv[1], RolloutConfig(backend="triton", timeout_s=.2,
                                               gpu_lock_timeout_s=float(sys.argv[2])))
print(json.dumps(dict(result=result, deadlines=deadlines)), flush=True)
'''


class GPULockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gpu-lock-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / 'task'
        self.task.mkdir()
        binary = self.root / 'nvidia-smi'
        binary.write_text('#!/bin/sh\nprintf "GPU-test-shared, Synthetic H200\\n"\n')
        binary.chmod(0o755)
        self.env = {**os.environ, 'CUDA_AGENT_DEVICE': 'cuda', 'CUDA_VISIBLE_DEVICES': '3,1',
                    'CUDA_AGENT_GPU_LOCK_DIR': str(self.root / 'locks'),
                    'PATH': str(self.root), 'PYTHONDONTWRITEBYTECODE': '1'}

    def process(self, code, *args):
        proc = subprocess.Popen([sys.executable, '-S', '-u', '-c', code, *args], cwd=ROOT,
                                env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=10)
        self.addCleanup(cleanup)
        return proc

    def line(self, proc):
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(timeout=10), 'child did not report readiness')
        return proc.stdout.readline().strip()

    def test_two_processes_wait_and_extend_rollout_deadline(self):
        holder = self.process(HOLDER)
        self.assertEqual(self.line(holder), 'HELD')
        waiter = self.process(WAITER, str(self.task), '8')
        # Compilation/verification complete while the other process holds the GPU.
        self.assertEqual(self.line(waiter), 'VERIFIED')
        time.sleep(.3)
        holder.stdin.write('x'); holder.stdin.flush()
        out, err = waiter.communicate(timeout=10)
        self.assertEqual(waiter.returncode, 0, err)
        data = json.loads(out)
        result, deadlines = data['result'], data['deadlines']
        self.assertEqual(result['reward/reason'], 'faster_than_both')
        wait = result['meta/gpu_lock_wait_s']
        self.assertGreaterEqual(wait, .25)
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertAlmostEqual(deadlines[2] - deadlines[1], wait, places=6)
        self.assertGreater(result['env/wall_s'], .2)
        self.assertEqual(result['meta/gpu_uuid'], 'GPU-test-shared')

    def test_sigkill_holder_releases_to_waiter(self):
        holder = self.process(HOLDER)
        self.assertEqual(self.line(holder), 'HELD')
        waiter = self.process(WAITER, str(self.task), '8')
        self.assertEqual(self.line(waiter), 'VERIFIED')
        time.sleep(.15)
        started = time.monotonic()
        holder.kill(); holder.wait(timeout=5)
        out, err = waiter.communicate(timeout=8)
        self.assertEqual(waiter.returncode, 0, err)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(json.loads(out)['result']['meta/gpu_lock'], 'held')

    def test_lock_wait_cap_is_gpu_busy_never_task_timeout(self):
        holder = self.process(HOLDER)
        self.assertEqual(self.line(holder), 'HELD')
        waiter = self.process(WAITER, str(self.task), '.3')
        self.assertEqual(self.line(waiter), 'VERIFIED')
        out, err = waiter.communicate(timeout=8)
        self.assertEqual(waiter.returncode, 0, err)
        data = json.loads(out)
        self.assertEqual(len(data['deadlines']), 2)
        self.assertEqual((data['result']['reward/reason'], data['result']['reward/value']), ('gpu_busy', -1))
        self.assertGreaterEqual(data['result']['meta/gpu_lock_wait_s'], .3)

    def test_all_measurement_stages_share_lock_and_release_on_failure(self):
        def stage(module, args, path, env, deadline):
            measurement = module not in ('utils.compile', 'utils.verification')
            with MeasurementLock(env, timeout_s=.03) as contender:
                self.assertEqual(contender.busy, measurement, module)
            if module == 'utils.verification':
                return 0, '[PASS] verify success', '', False
            if module == 'utils.profiling':
                return 0, json.dumps(dict(t_eager_us=100, t_compile_us=80, t_gen_us=50, provenance=1)), '', False
            return 0, '{}', '', False
        with patch.dict(os.environ, self.env, clear=True), patch('utils.rollout_env._stage', side_effect=stage):
            result = run_task(str(self.task), RolloutConfig(backend='triton', benchmark=True, diagnostics=True))
        self.assertEqual(result['meta/gpu_lock_wait_s'], 0.)
        self.assertEqual(result['reward/value'], 3)
        for failure in ((1, '', 'crash', False), (None, '', '', True), ValueError('bad profile')):
            outcomes = [(0, '', '', False), (0, '[PASS] verify success', '', False), failure]
            with patch.dict(os.environ, self.env, clear=True), patch('utils.rollout_env._stage', side_effect=outcomes):
                result = run_task(str(self.task), RolloutConfig(backend='triton'))
            self.assertEqual(result['reward/reason'], 'timeout' if isinstance(failure, tuple) and failure[3] else 'profiling')
            with MeasurementLock(self.env, timeout_s=.03) as lock:
                self.assertTrue(lock.acquired)

    def test_cpu_opt_out_and_unknown_uuid(self):
        for env in ({**self.env, 'CUDA_AGENT_DEVICE': 'cpu'}, {**self.env, 'CUDA_VISIBLE_DEVICES': ''}):
            with patch('utils.gpu_lock.subprocess.run') as probe, MeasurementLock(env) as lock:
                self.assertEqual(lock.metadata['meta/gpu_lock'], 'none_cpu')
                self.assertEqual(lock.metadata['meta/gpu_lock_wait_s'], 0.)
                probe.assert_not_called()
        with MeasurementLock(self.env, enabled=False) as lock:
            self.assertEqual(lock.metadata['meta/gpu_lock'], 'disabled')
            self.assertFalse(lock.acquired)
        missing = {**self.env, 'PATH': str(self.root / 'missing')}
        with MeasurementLock(missing) as lock:
            self.assertTrue(lock.acquired)
            self.assertEqual(lock.metadata['meta/gpu_lock'], 'unknown_uuid')
            self.assertIsNone(lock.metadata['meta/gpu_uuid'])
            self.assertIn('unavailable', lock.metadata['meta/gpu_uuid_reason'])
        cpu = {k: v for k, v in missing.items() if k not in ('CUDA_AGENT_DEVICE', 'CUDA_VISIBLE_DEVICES')}
        with patch('utils.gpu_lock._has_cuda_driver', return_value=False):
            self.assertEqual(device_identity(cpu)['meta/gpu_lock'], 'none_cpu')
        with patch('utils.gpu_lock._has_cuda_driver', return_value=True):
            self.assertEqual(device_identity(cpu)['meta/gpu_lock'], 'unknown_uuid')

    def test_visible_device_mapping_uuid_keys_and_bad_output(self):
        for device, visible, selected in [('cuda', '3,1', '3'), ('cuda:1', '3,1', '1'),
                                           ('cuda', 'GPU-visible', 'GPU-visible'), ('cuda:1', None, '1')]:
            env = {**self.env, 'CUDA_AGENT_DEVICE': device}
            if visible is None:
                env.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                env['CUDA_VISIBLE_DEVICES'] = visible
            with patch('utils.gpu_lock.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'GPU-test, Fake\n')) as probe:
                info = device_identity(env)
            self.assertEqual(probe.call_args.args[0][-1], selected)
            self.assertEqual(info['meta/gpu_lock_key'], 'GPU-test')
        with patch('utils.gpu_lock.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'bad output')):
            self.assertEqual(device_identity(self.env)['meta/gpu_lock'], 'unknown_uuid')
        with patch('utils.gpu_lock.subprocess.run', side_effect=subprocess.TimeoutExpired('probe', 5)):
            info = device_identity({**self.env, 'CUDA_VISIBLE_DEVICES': 'GPU-visible'})
        self.assertEqual(info['meta/gpu_lock_key'], 'GPU-visible')
        self.assertIsNone(info['meta/gpu_uuid'])
        # Different visibility lists resolving to the same UUID contend on one inode.
        with MeasurementLock(self.env) as first, MeasurementLock({**self.env, 'CUDA_VISIBLE_DEVICES': '7'}, timeout_s=.03) as second:
            self.assertTrue(first.acquired)
            self.assertTrue(second.busy)

    def test_config_flags_validation_and_torch_free_import(self):
        parser = argparse.ArgumentParser()
        RolloutConfig.add_args(parser)
        cfg = RolloutConfig.from_args(parser.parse_args(['--no-gpu-lock', '--gpu-lock-timeout-s', '.25']))
        self.assertFalse(cfg.gpu_lock)
        self.assertEqual(cfg.gpu_lock_timeout_s, .25)
        for value in (0, -1, float('nan'), float('inf'), True, 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RolloutConfig(gpu_lock_timeout_s=value)
        with self.assertRaises(ValueError):
            RolloutConfig(gpu_lock='yes')
        subprocess.run([sys.executable, '-S', '-c',
                        'import sys; from utils import gpu_lock,rollout_env; assert "torch" not in sys.modules'],
                       cwd=ROOT, check=True, timeout=10)


class BusyAdvantageTests(unittest.TestCase):
    def test_busy_task_suite_stops_before_supplemental_diagnostics(self):
        from utils.task_catalog import select_cases
        from utils.task_suite import evaluate_case
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'case'
            def fake_run(*args, **kwargs):
                return {'env/verify_ok': True, 'reward/value': -1., 'reward/base': -1,
                        'reward/reason': 'gpu_busy', 'meta/gpu_lock_wait_s': .25}
            with patch('utils.task_suite.cuda_unavailable', return_value=None), \
                    patch('utils.task_suite.available_backends', return_value={'triton': None}), \
                    patch('utils.task_suite.run_task', side_effect=fake_run), \
                    patch('utils.task_suite._stage') as stage:
                case = select_cases('lora_fused_sgmv', 'smoke')[0]
                result = evaluate_case('lora_fused_sgmv', case.id, 'triton', 'honest', target)
            stage.assert_not_called()
            self.assertEqual((result['suite/status'], result['suite/reason']), ('unavailable', 'gpu_busy'))
            self.assertEqual(result['reward/reason'], 'gpu_busy')

    def test_only_gpu_busy_is_excluded_with_original_group_boundaries(self):
        reasons = ('compile', 'verification', 'profiling', 'timeout', 'provenance',
                   'max_turns', 'unmeasured', 'timing', 'correctness', 'diagnostic_profile')
        for algo in ('grpo', 'dapo', 'cispo'):
            cfg = dataclasses.replace(RolloutConfig.preset(algo), group_size=3, adv_norm='none')
            for reason in reasons:
                rows = [mock_rollout(3), mock_rollout(-1, reason), mock_rollout(-1, 'gpu_busy')]
                pairs, stats = build_batch(['task'] * 3, rows, cfg)
                self.assertEqual(pairs, [('0', 2), ('1', -2)], (algo, reason))
                self.assertEqual(stats['adv/gpu_busy_excluded'], 1)
                self.assertEqual(stats['reward/mean'], 1/3)
                normal = rows[:2] + [mock_rollout(1)]
                old_pairs, normal_stats = build_batch(['task'] * 3, normal, cfg)
                self.assertEqual([a for _, a in old_pairs], group_advantages([3, -1, 1], 'none'))
                self.assertEqual(normal_stats['adv/gpu_busy_excluded'], 0)
            rows = [mock_rollout(3), mock_rollout(-1, 'gpu_busy'), mock_rollout(-1, 'gpu_busy'),
                    mock_rollout(-1, 'gpu_busy'), mock_rollout(1), mock_rollout(3)]
            pairs, stats = build_batch(['task'] * 6, rows, cfg)
            self.assertEqual(pairs, ([('0', 0)] if algo == 'grpo' else []) + [('4', -1), ('5', 1)])
            self.assertEqual(stats['adv/gpu_busy_excluded'], 3)
            pairs, stats = build_batch(['task'] * 3, [mock_rollout(-1, 'gpu_busy')] * 3, cfg)
            self.assertEqual(pairs, [])
            self.assertEqual(stats['adv/gpu_busy_excluded'], 3)
