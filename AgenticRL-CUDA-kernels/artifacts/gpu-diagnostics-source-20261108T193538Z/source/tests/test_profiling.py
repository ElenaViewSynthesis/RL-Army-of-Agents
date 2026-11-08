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
        primers = [FakeProfiler([]), FakeProfiler([])] if kwargs.get('prime', True) else []
        with patch.dict(sys.modules, torch=fake_torch), patch('utils.profiling.get_prof_ctx',
                side_effect=primers + [FakeProfiler(e) for e in events]):
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
                metadata = key.startswith('profile/') or key == 'timing_reason'
                self.assertEqual(classify('profile',key), 'artifact' if metadata else 'table')
                names_artifact = metadata or key in ('t_eager_us','t_compile_us')
                self.assertEqual(classify('names','backends/triton/profile/'+key),
                                 'artifact' if names_artifact else 'table')

    def test_all_empty_and_success(self):
        profile = self.capture([[],[],[]])
        self.assertEqual(profile['timing_reason'],'empty_profile:gen')
        self.assertIsNone(profile['provenance'])
        self.assertTrue(all(profile['t_'+s+'_us'] is None for s in ('eager','compile','gen')))
        profile = self.capture([[event()]]*3)
        self.assertTrue(profile['timing_measured'])
        self.assertIsNone(profile['timing_reason'])
        self.assertFalse(profile['profile/diagnostic'])

    def test_nonpositive_and_invalid_sessions_fail_closed_without_losing_samples(self):
        for index, session in enumerate(('gen', 'eager', 'compile')):
            for duration, reason in ((0.0, 'nonpositive_profile'), (-1.0, 'invalid_profile'),
                                     (float('nan'), 'invalid_profile'),
                                     (float('inf'), 'invalid_profile')):
                with self.subTest(session=session, duration=duration):
                    events = [[event()]] * 3
                    events[index] = [event(duration=duration)]
                    profile = self.capture(events)
                    self.assertFalse(profile['timing_measured'])
                    self.assertEqual(profile['timing_reason'], reason + ':' + session)
                    self.assertIsNone(profile['t_' + session + '_us'])
                    self.assertEqual(profile['profile/' + session + '_events'], 1)
                    sample = json.loads(profile['profile/' + session + '_kernel_samples'])[0]
                    self.assertEqual(sample[0], 'axpby_kernel')
                    self.assertEqual(sample[1], str(duration) if reason == 'invalid_profile'
                                     and duration not in (-1.0,) else duration)
                    json.dumps(profile, allow_nan=False)
                    if session == 'gen':
                        self.assertIsNone(profile['provenance'])
                    else:
                        self.assertEqual(profile['provenance'], 1.0)
                    with scratch() as tmp:
                        task = make_task(tmp)
                        stages = [(0, 'ok', '', False), (0, '[PASS] verify success', '', False),
                                  (0, json.dumps(profile), '', False)]
                        with patch('utils.rollout_env._stage', side_effect=stages) as stage:
                            result = run_task(task, RolloutConfig())
                        self.assertEqual(stage.call_count, 3)
                        self.assertEqual((result['reward/value'], result['reward/reason']),
                                         (-1, reason + ':' + session))

    def test_valid_samples_preserve_cuda_events_only(self):
        profile = self.capture([[event(), event(device='CPU', duration=100)]] * 3)
        for session in ('gen', 'eager', 'compile'):
            self.assertEqual(json.loads(profile['profile/' + session + '_kernel_samples']),
                             [['axpby_kernel', 2]])
            self.assertEqual(profile['t_' + session + '_us'], 2)

    def test_conditioning_precedes_every_model_and_unprimed_is_diagnostic(self):
        from collections import Counter
        calls = Counter()
        contexts = []
        def model(session):
            def forward():
                calls[session] += 1
            return forward
        def profiler():
            contexts.append(dict(calls))
            return FakeProfiler([] if len(contexts) <= 2 else [event()])
        fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext,
                                         cuda=types.SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, torch=fake_torch), patch('utils.profiling.get_prof_ctx', profiler):
            profile = profile_models({s: model(s) for s in ('gen', 'eager', 'compile')},
                                     {s: [] for s in ('gen', 'eager', 'compile')}, 'triton', 1,
                                     agent_names=['axpby_kernel'])
        self.assertEqual(contexts[:2], [{}, {}])
        self.assertEqual(len(contexts), 5)
        self.assertEqual(calls, Counter(dict(gen=21, eager=21, compile=21)))
        self.assertEqual(json.loads(profile['profile/primer_event_counts']), [0, 0])
        self.assertEqual(profile['profile/timing_method'], 'cuda-leaf-sum-two-empty-prime-v1')
        self.assertFalse(profile['profile/diagnostic'])
        unprimed = self.capture([[event()]] * 3, prime=False)
        self.assertTrue(unprimed['profile/diagnostic'])
        self.assertEqual(unprimed['profile/timing_method'], 'cuda-leaf-sum-unprimed-v1')
        self.assertEqual(json.loads(unprimed['profile/primer_event_counts']), [])


class DiagnosticTests(unittest.TestCase):
    def test_order_experiment_is_complete_and_counterbalanced(self):
        from collections import Counter
        from scripts.diagnose_profile_order import plan
        from utils.profile_diagnostic import ORDERS, PREPARATIONS
        jobs = plan(['triton', 'cutedsl'], 4)
        self.assertEqual(len(jobs), 64)
        self.assertEqual(len(set(jobs)), 64)
        counts = Counter((backend, preparation, order) for backend, _, preparation, order in jobs)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(set(counts), {(b, p, o) for b in ('triton', 'cutedsl')
                                      for p in PREPARATIONS for o in ORDERS})
        self.assertEqual(jobs[0][2:], (PREPARATIONS[0], ORDERS[0]))
        self.assertEqual(jobs[16][2:], (PREPARATIONS[-1], ORDERS[-1]))
        extended = plan(['triton', 'cutedsl'], 4, ('double-empty-prime', 'triple-empty-prime'))
        self.assertEqual(len(extended), 32)
        self.assertEqual(set(Counter((b, p, o) for b, _, p, o in extended).values()), {4})

    def test_preparation_controls_are_separate_and_always_diagnostic(self):
        from collections import Counter
        from utils import profile_diagnostic
        for preparation in profile_diagnostic.ALL_PREPARATIONS:
            with self.subTest(preparation=preparation):
                calls = Counter()
                def model(name):
                    def forward():
                        calls[name] += 1
                    return forward
                eager, generated, compiled = (model(s) for s in ('eager', 'gen', 'compile'))
                fake_torch = types.SimpleNamespace(
                    manual_seed=lambda seed: None, no_grad=contextlib.nullcontext,
                    compile=lambda model: compiled,
                    cuda=types.SimpleNamespace(synchronize=lambda: None))
                with (patch.dict(sys.modules, torch=fake_torch),
                      patch.object(profile_diagnostic, 'initialize_models',
                                   return_value=(eager, generated, [], [])),
                      patch.object(profile_diagnostic, 'agent_kernel_names', return_value=['authored']),
                      patch.object(profile_diagnostic, 'get_prof_ctx',
                                   return_value=FakeProfiler([])) as empty,
                      patch.object(profile_diagnostic, 'benchmark_model',
                                   return_value=(2.0, [('eager_kernel', 2.0)])) as primer,
                      patch.object(profile_diagnostic, 'profile_models',
                                   return_value={'profile/diagnostic': False}) as measured):
                    result = profile_diagnostic.capture('triton', 'gen,eager,compile', preparation)
                self.assertTrue(result['diagnostic_only'])
                self.assertTrue(result['profile']['profile/diagnostic'])
                measured.assert_called_once()
                self.assertEqual(measured.call_args.kwargs['order'], 'gen,eager,compile')
                self.assertFalse(measured.call_args.kwargs['prime'])
                self.assertEqual(calls, Counter(dict(eager=20, gen=20, compile=20))
                                 if preparation == 'warmup-all' else Counter())
                count = {'empty-prime': 1, 'double-empty-prime': 2,
                         'triple-empty-prime': 3}.get(preparation, 0)
                self.assertEqual(empty.call_count, count)
                self.assertEqual(primer.call_count, int(preparation == 'eager-prime'))
                if preparation == 'eager-prime':
                    primer.assert_called_once_with(eager, [], 20, 10, True)
                if preparation == 'empty-prime':
                    self.assertEqual(result['primer'], {'intentional_empty': True, 'cuda_events': 0})
                elif count:
                    self.assertEqual(result['primer'], {'intentional_empty': True, 'cuda_events': 0,
                                                       'empty_capture_events': [0] * count})
