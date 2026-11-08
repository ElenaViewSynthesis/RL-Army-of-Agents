"""Offline acceptance and diagnostic-plan tests; never contact Modal."""
import copy
import io
import json
import unittest
from pathlib import Path

from cloud.acceptance import review
from scripts.diagnose_timing import plan
from tests.support import ROOT


class AcceptanceTests(unittest.TestCase):
    def evidence(self):
        from utils.backend import BACKENDS
        selected = list(BACKENDS)
        results = {'rollouts':{},'profiles':{}}
        profile = {'timing_measured':True,'profile/diagnostic':False,
                   **{f't_{s}_us':1.5 for s in ('eager','compile','gen')},
                   **{f'profile/{s}_events':10 for s in ('eager','compile','gen')}}
        for backend in selected:
            for case, provenance, reward, reason in ((backend,1.0,2,'faster_than_eager'),
                    ('reward-hack-torchop' if backend=='triton' else 'reward-hack-torchop-'+backend,0.0,-1,'provenance')):
                results['rollouts'][case] = {'env/verify_ok':True,'env/provenance':provenance,
                    **{f'env/t_{s}_us':1.5 for s in ('eager','compile','gen')},
                    'reward/value':reward,'reward/base':reward,'reward/reason':reason}
                results['profiles'][case] = copy.deepcopy(profile)
        names = {'backends':{b:{'measured':True,'provenance':1.0,'unmatched_agent_names':[],
                                 'profile':profile} for b in selected}}
        probe = {'runtime/gpu0_uuid':'gpu','runtime/driver_version':'test'}
        return ['synthetic',selected,[{'returncode':0,'timed_out':False}],results,names,
                dict(probe,run_id='synthetic'),probe,profile]

    def test_positive_acceptance_and_expected_failures(self):
        args = self.evidence()
        self.assertEqual(review(*args)['status'],'pass')
        for mutation in ('zero','empty','unmatched','probe','missing_hack','stage','missing_event_count'):
            with self.subTest(mutation=mutation):
                args = self.evidence()
                if mutation=='zero':args[3]['rollouts']['cutedsl']['env/t_eager_us']=0.0
                if mutation=='empty':args[3]['profiles']['cutedsl']['timing_measured']=False
                if mutation=='unmatched':args[4]['backends']['cutile']['unmatched_agent_names']=['missed']
                if mutation=='probe':args[5]['runtime/gpu0_uuid']='stale'
                if mutation=='missing_hack':del args[3]['rollouts']['reward-hack-torchop-cuda']
                if mutation=='stage':args[2][0]['returncode']=1
                if mutation=='missing_event_count':del args[3]['profiles']['cutedsl']['profile/eager_events']
                self.assertEqual(review(*args)['status'],'partial')

    def test_historical_zero_baseline_cannot_be_accepted(self):
        run_id='20261108T014026Z'
        def load(name):return json.loads((ROOT/'artifacts'/f'{name}-{run_id}.json').read_text())
        r=review(run_id,['cuda','triton','tilelang','cutile','cutedsl'],[{'returncode':0}],
                 load('results'),load('kernel_names'),load('environment'),load('environment'),{})
        self.assertEqual(r['status'],'partial')
        self.assertIn({'case':'cutedsl','key':'env/t_eager_us','value':0.0},r['timing_anomalies'])


class PlanTests(unittest.TestCase):
    def test_default_counts_and_balanced_orders(self):
        from collections import Counter
        from utils.backend import BACKENDS
        jobs=plan(BACKENDS)
        self.assertEqual(Counter(b for b,_,_ in jobs),dict(cuda=20,triton=20,tilelang=20,cutile=20,cutedsl=100))
        for b in BACKENDS:
            counts=Counter(order for backend,_,order in jobs if backend==b)
            self.assertEqual(len(set(counts.values())),1)
        self.assertEqual(len(set((b,i) for b,i,_ in jobs)),180)
        self.assertEqual(len(plan(['cutedsl'],4)),4)


class RunnerTests(unittest.TestCase):
    def runner(self):
        import importlib.util
        import types
        from unittest.mock import patch
        class Image:
            def __getattr__(self,name):return lambda *a,**kw:self
        class App:
            def __init__(self,*a,**kw):pass
            def function(self,**kw):return lambda fn:fn
            def local_entrypoint(self):return lambda fn:fn
        fake = types.SimpleNamespace(Image=types.SimpleNamespace(from_registry=lambda *a,**kw:Image()),
            Volume=types.SimpleNamespace(from_name=lambda *a,**kw:types.SimpleNamespace(commit=lambda:None)),App=App)
        spec=importlib.util.spec_from_file_location('test_modal_runtime',ROOT/'cloud/modal_runtime.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',modal=fake):spec.loader.exec_module(module)
        return module

    def test_forwarding_is_opt_in_and_allowlisted(self):
        from unittest.mock import patch
        runner=self.runner()
        with patch.dict('os.environ',{'TRITON_KERNEL_DUMP':'1','CUTE_DSL_KEEP':'ir,ptx','SECRET_TOKEN':'never-forward'}):
            self.assertEqual(runner.forwarded_environment(False),{})
            actual=runner.forwarded_environment(True)
        self.assertEqual(actual['TRITON_KERNEL_DUMP'],'1')
        self.assertEqual(actual['CUTE_DSL_KEEP'],'ir,ptx')
        self.assertNotIn('SECRET_TOKEN',actual)
        from unittest.mock import Mock
        call=Mock(object_id='fc-synthetic')
        call.get.side_effect=[TimeoutError(), {'done':True}]
        function=Mock()
        function.spawn.return_value=call
        self.assertEqual(runner.await_job(function,'job'),{'done':True})
        function.spawn.assert_called_once_with('job')
        self.assertEqual(call.get.call_count,2)
        self.assertTrue(all(c.kwargs=={'timeout':30} for c in call.get.call_args_list))

        calls=[]
        def run(*args):
            calls.append(args)
            return {'returncode':0}
        with patch.object(runner,'_run',side_effect=run):
            runner.diagnose_timing('synthetic','cutedsl',2,actual)
        self.assertIn('--repeat',calls[1][0])
        self.assertEqual(calls[1][0][-1],'2')
        self.assertEqual(calls[1][-1],actual)


    def test_wait_for_call_retries_only_builtin_timeout(self):
        from unittest.mock import Mock, patch
        from cloud.modal_job import wait_for_call
        call=Mock(object_id='fc-synthetic')
        call.get.side_effect=[TimeoutError(),TimeoutError(),{'done':True}]
        with patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(wait_for_call(call),{'done':True})
        self.assertEqual(json.loads(out.getvalue()),{'function_call_id':'fc-synthetic'})
        self.assertEqual(call.get.call_count,3)
        self.assertTrue(all(c.kwargs=={'timeout':30} for c in call.get.call_args_list))

    def test_non_timeout_error_reports_recovery_and_never_respawns(self):
        from unittest.mock import Mock, patch
        runner=self.runner()
        for raised in (RuntimeError('stand-in credential material'),type('PollTimeout',(TimeoutError,),{})()):
            call=Mock(object_id='fc-synthetic')
            call.get.side_effect=raised
            function=Mock()
            function.spawn.return_value=call
            with patch('sys.stdout',new_callable=io.StringIO),patch('sys.stderr',new_callable=io.StringIO) as err:
                with self.assertRaises(type(raised)):
                    runner.await_job(function,'job')
            function.spawn.assert_called_once_with('job')
            self.assertEqual(call.get.call_count,1)
            record=json.loads(err.getvalue())
            self.assertEqual(record['function_call_id'],'fc-synthetic')
            self.assertEqual(record['error_type'],type(raised).__name__)
            self.assertEqual(record['recovery'],'python -m cloud.modal_job --call-id fc-synthetic')
            self.assertNotIn('credential material',err.getvalue())

    def test_resume_cli_polls_call_id_without_runner(self):
        import types
        from unittest.mock import Mock, patch
        from cloud import modal_job
        unhydrated=Mock(spec=['hydrate','get'])
        hydrated=Mock(object_id='fc-AbC123')
        hydrated.get.return_value={'run_id':'resumed'}
        unhydrated.hydrate.return_value=hydrated
        from_id=Mock(return_value=unhydrated)
        fake=types.SimpleNamespace(FunctionCall=types.SimpleNamespace(from_id=from_id))
        with patch.dict('sys.modules',{'modal':fake,'cloud.modal_runtime':None}),\
             patch('sys.argv',['modal_job','--call-id','fc-AbC123']),\
             patch('sys.stdout',new_callable=io.StringIO) as out:
            modal_job.main()
        from_id.assert_called_once_with('fc-AbC123')
        unhydrated.hydrate.assert_called_once_with()
        unhydrated.get.assert_not_called()
        hydrated.get.assert_called_once_with(timeout=30)
        text=out.getvalue()
        self.assertEqual(json.loads(text.splitlines()[0]),{'function_call_id':'fc-AbC123'})
        self.assertEqual(json.loads(text[text.index('\n'):]),{'run_id':'resumed'})

    def test_resume_cli_rejects_invalid_call_id_before_modal_import(self):
        from unittest.mock import patch
        from cloud import modal_job
        with patch.dict('sys.modules',{'modal':None}),\
             patch('sys.argv',['modal_job','--call-id','fc-abc extra']),\
             patch('sys.stderr',new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                modal_job.main()

    def test_main_preflight_uses_spawn_helper_not_remote(self):
        from unittest.mock import Mock, patch
        runner=self.runner()
        call=Mock(object_id='fc-pre');call.get.return_value={'runtime/ok':True}
        tests_call=Mock(object_id='fc-tests');tests_call.get.return_value={'acceptance':'pass'}
        preflight=Mock();preflight.spawn.return_value=call
        gpu_fn=Mock();gpu_fn.spawn.return_value=tests_call
        with patch.object(runner,'preflight',preflight),patch.object(runner,'gpu_tests',gpu_fn),\
             patch('sys.stdout',new_callable=io.StringIO):
            runner.main(tests=True)
        preflight.spawn.assert_called_once()
        preflight.remote.assert_not_called()
        gpu_fn.spawn.assert_called_once()
        self.assertTrue(all(c.kwargs=={'timeout':30} for c in call.get.call_args_list+tests_call.get.call_args_list))

    def test_main_routes_diagnose_order_exclusively(self):
        from unittest.mock import Mock, patch
        runner=self.runner()
        preflight=Mock();preflight.spawn.return_value=Mock(object_id='fc-pre',get=Mock(return_value={'ok':1}))
        order=Mock();order.spawn.return_value=Mock(object_id='fc-order',get=Mock(return_value={'runs':2}))
        gpu_fn,diag=Mock(),Mock()
        with patch.object(runner,'preflight',preflight),patch.object(runner,'diagnose_order',order),\
             patch.object(runner,'gpu_tests',gpu_fn),patch.object(runner,'diagnose_timing',diag),\
             patch('sys.stdout',new_callable=io.StringIO):
            runner.main(diagnose_order_test=True)
            self.assertEqual(order.spawn.call_args.args[1:],('triton,cutedsl',4))
            runner.main(diagnose_order_test=True,backends='cutile',repeat=2)
            self.assertEqual(order.spawn.call_args.args[1:],('cutile',2))
            for flags in ({'tests':True,'diagnose_order_test':True},
                          {'diagnose':True,'diagnose_order_test':True},
                          {'tests':True,'diagnose':True}):
                with self.assertRaises(ValueError):runner.main(**flags)
        self.assertEqual(order.spawn.call_count,2)
        self.assertEqual(preflight.spawn.call_count,2)
        gpu_fn.spawn.assert_not_called()
        diag.spawn.assert_not_called()


    def test_runner_writes_automatic_review_from_fresh_outputs(self):
        from unittest.mock import patch
        from tests.support import scratch
        runner=self.runner()
        args=AcceptanceTests().evidence()
        run_id,selected,_,results,names,_,probe,profile=args
        with scratch() as tmp:
            root=Path(tmp);remote=root/'repo';(remote/'artifacts').mkdir(parents=True)
            (remote/'artifacts/results.json').write_text('stale')
            out=root/'volume'/run_id;out.mkdir(parents=True)
            def mapped(path):return root/'volume' if path=='/artifacts' else Path(path)
            def run(argv,name,timeout,runid,env):
                if name=='require':(out/'require.stdout.log').write_text(json.dumps(probe))
                if name=='stock_pipeline':(out/'stock_pipeline.stdout.log').write_text(json.dumps(profile))
                if name=='kernel_names':(remote/'artifacts/kernel_names.json').write_text(json.dumps(names))
                if name=='validate':(remote/'artifacts/results.json').write_text(json.dumps(results))
                self.assertIn('CUDA_AGENT_BENCH_SAMPLES_DIR',env)
                return {'cmd':'synthetic','returncode':0,'timed_out':False}
            with patch.object(runner,'Path',side_effect=mapped),patch.object(runner,'REMOTE',str(remote)),patch.object(runner,'_run',side_effect=run):
                result=runner.gpu_tests(run_id)
            review_record=json.loads((out/f'acceptance-review-{run_id}.json').read_text())
            self.assertEqual(review_record['status'],'pass')
            self.assertEqual(result['acceptance'],'pass')
            self.assertTrue(review_record['environment_matches_require_probe'])
            self.assertEqual(json.loads((out/'environment.json').read_text())['runtime/gpu0_uuid'],'gpu')
