"""Synthetic profiler captures, including empty CUDA sessions; no GPU required."""
import contextlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import make_task, scratch
from utils.config import RolloutConfig
from utils.profiling import profile_models
from utils.rollout_env import run_task


class FakeProfiler:
    def __init__(self, events):
        self.captured = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def events(self):
        return self.captured

    def export_chrome_trace(self, path):
        Path(path).write_text(json.dumps({'traceEvents': []}))


def event(name='axpby_kernel', device='CUDA', duration=2):
    return types.SimpleNamespace(name=name, device_type=types.SimpleNamespace(name=device), device_time=duration)


class CaptureTests(unittest.TestCase):
    def capture(self, events, **kwargs):
        fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext,
                                         cuda=types.SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, torch=fake_torch), patch('utils.profiling.get_prof_ctx',
                side_effect=[FakeProfiler(e) for e in events]):
            return profile_models({s: lambda: None for s in ('gen','eager','compile')},
                                  {s: [] for s in ('gen','eager','compile')}, 'triton', 1,
                                  agent_names=['axpby_kernel'], **kwargs)

    def test_empty_session_fails_closed_and_preserves_evidence(self):
        for i, session in enumerate(('gen','eager','compile')):
            with self.subTest(session=session):
                events = [[event()], [event('aten::add')], [event('triton_poi_fused_add_0')]]
                events[i] = [event(device='CPU')]  # CPU scopes must not rescue an empty CUDA capture.
                profile = self.capture(events)
                self.assertFalse(profile['timing_measured'])
                self.assertEqual(profile['timing_reason'], 'empty_profile:'+session)
                self.assertIsNone(profile['t_'+session+'_us'])
                self.assertEqual(profile['profile/'+session+'_events'], 0)
                with scratch() as tmp:
                    task = make_task(tmp)
                    stages = [(0,'ok','',False), (0,'[PASS] verify success','',False),
                              (0,json.dumps(profile),'',False)]
                    sink = {}
                    with patch('utils.rollout_env._stage', side_effect=stages) as stage:
                        result = run_task(task, RolloutConfig(), profile_out=sink)
                    self.assertEqual(stage.call_count, 3)  # No retry.
                    self.assertEqual((result['reward/value'],result['reward/reason']),
                                     (-1, 'empty_profile:'+session))
                    self.assertEqual(len(result), 18)
                    self.assertEqual(sink, profile)

    def test_counts_names_traces_order_and_contract(self):
        with scratch() as tmp:
            profile = self.capture([[event('aten::mul'),event('aten::add')], [event()], [event()]],
                                   order='eager,compile,gen', trace_dir=tmp)
            self.assertTrue(profile['timing_measured'])
            self.assertTrue(profile['profile/diagnostic'])
            self.assertEqual(profile['profile/eager_events'],2)
            self.assertEqual(json.loads(profile['profile/eager_kernel_names']), ['aten::add','aten::mul'])
            for session in ('gen','eager','compile'):
                self.assertTrue(Path(profile['profile/'+session+'_trace']).is_file())
            task = make_task(tmp)
            stages = [(0,'ok','',False), (0,'[PASS] verify success','',False),
                      (0,json.dumps(profile),'',False)]
            with patch('utils.rollout_env._stage',side_effect=stages):
                result = run_task(task,RolloutConfig())
            self.assertEqual((result['reward/value'],result['reward/reason']),(-1,'diagnostic_profile'))
            from tracking.contract import classify
            for key in profile:
                self.assertEqual(classify('profile',key), 'table')
                self.assertEqual(classify('names','backends/triton/profile/'+key), 'table')

    def test_all_empty_and_success(self):
        profile = self.capture([[],[],[]])
        self.assertEqual(profile['timing_reason'],'empty_profile:gen')
        self.assertIsNone(profile['provenance'])
        self.assertTrue(all(profile['t_'+s+'_us'] is None for s in ('eager','compile','gen')))
        profile = self.capture([[event()]]*3)
        self.assertTrue(profile['timing_measured'])
        self.assertIsNone(profile['timing_reason'])
        self.assertFalse(profile['profile/diagnostic'])
