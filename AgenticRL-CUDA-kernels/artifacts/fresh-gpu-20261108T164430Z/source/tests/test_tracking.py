"""Tracking tests: fake wandb/weave only; no live service is ever contacted."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.support import ROOT, assert_flat, make_task, scratch
from tracking import wandb_log, weave_trace

ARTIFACTS = ROOT / "artifacts"
RUN_OK, RUN_FAILED, RUN_BUILD = ("20261106T185236Z", "20261106T142830Z",
                                 "build-im-u7ElhT76HK3e4MiGfUc5JZ")


def have(run_id):
    return (ARTIFACTS / f"results-{run_id}.json").is_file()


class FakeRun:
    _ATTRS = {"logged", "artifacts", "summary", "definitions", "url", "id", "finished"}

    def __init__(self):
        self.logged, self.artifacts, self.summary = [], [], {}
        self.definitions = []
        self.url, self.id, self.finished = "https://fake/run", "rid", None

    def __setattr__(self, name, value):
        # Like wandb's Run, which raises for any attribute it does not define.
        if name not in self._ATTRS:
            raise Exception(f"Attribute {name} is not supported on Run object.")
        super().__setattr__(name, value)

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, row, step=None):
        self.logged.append((row, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.finished = exit_code


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path, **kwargs):
        self.files.append(path)


class FakeWandbModule:
    def __init__(self):
        self.init_kwargs, self.run = None, FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    Table = staticmethod(lambda columns, data: ("table", columns, data))
    Artifact = FakeArtifact


class PureHelpersTests(unittest.TestCase):
    def test_metrics_and_config(self):
        result = {"env/verify_ok": True, "reward/value": 2.0, "log/stderr_tail": "x",
                  "artifact/trace": "/p", "bench/available": False, "adv/mean": 0.0}
        self.assertEqual(wandb_log.metrics_for_wandb(result), {"adv/mean": 0.0})
        config = wandb_log.config_for_wandb({"iters": 10},
                                            {"runtime/driver_ok": True, "other": 1},
                                            source_command="cmd")
        self.assertEqual(config, {"cfg/iters": 10, "runtime/driver_ok": True,
                                  "source_command": "cmd"})


@unittest.skipUnless(have(RUN_OK), "H200 artifacts not present")
class BackfillPayloadTests(unittest.TestCase):
    def payload(self, run_id):
        return wandb_log.backfill_payload(run_id, ARTIFACTS)

    def test_finished_run(self):
        p = self.payload(RUN_OK)
        self.assertEqual(json.loads(json.dumps(p, allow_nan=False)), p)
        self.assertEqual(len(p["rollouts"]), 12)
        self.assertEqual(len(p["kernels"]), 5)
        self.assertTrue(all(r["class"] == "agent" for r in p["kernels"]))
        self.assertEqual((p["summary"]["gpu_tests"], p["summary"]["unmatched_agent_names_total"]),
                         ("OK", 0))
        # overlong also passes provenance and earns >0: 6 of each, not 5.
        self.assertEqual((p["summary"]["provenance_pass"], p["summary"]["reward_positive"]), (6, 6))
        self.assertEqual(p["tags"], ["gpu-tests", "finished", "h200", "backfill"])
        assert_flat(self, p["summary"])

    def test_failed_run(self):
        p = self.payload(RUN_FAILED)
        self.assertEqual(p["summary"]["gpu_tests"], "FAILED")
        cuda = next(r for r in p["kernels"] if r["backend"] == "cuda")
        self.assertFalse(cuda["measured"])
        self.assertIsNone(cuda["event"])

    def test_build_run_and_unknown(self):
        p = self.payload(RUN_BUILD)
        self.assertEqual(p["status"], "failed")
        self.assertEqual(p["rollouts"], [])
        self.assertTrue(p["files"])
        with self.assertRaises(ValueError):
            self.payload("no-such-run")

    def test_dry_run_all_covers_manifest(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            wandb_log.main(["backfill", "--all", "--dry-run", "--artifacts", str(ARTIFACTS)])
        payloads = json.loads(out.getvalue())
        self.assertEqual(len(payloads), 5)


@unittest.skipUnless(have(RUN_OK), "H200 artifacts not present")
class LogBackfillTests(unittest.TestCase):
    def test_init_log_summary_finish_calls(self):
        wandb = FakeWandbModule()
        url = wandb_log.log_backfill(wandb_log.backfill_payload(RUN_OK, ARTIFACTS),
                                     wandb_module=wandb)
        self.assertEqual(url, "https://fake/run")
        kw = wandb.init_kwargs
        self.assertEqual((kw["id"], kw["job_type"], kw["tags"], kw["resume"]),
                         ("h200-20261106T185236Z", "gpu-tests",
                          ["gpu-tests", "finished", "h200", "backfill"], "allow"))
        self.assertEqual(len(wandb.run.logged), 1)   # tables + aggregate batch stats only
        self.assertEqual(wandb.run.finished, 0)
        wandb2 = FakeWandbModule()
        wandb_log.log_backfill(wandb_log.backfill_payload(RUN_FAILED, ARTIFACTS),
                               wandb_module=wandb2)
        self.assertEqual(wandb2.run.finished, 1)

    def test_new_run_id_never_resumes_with_unique_artifact(self):
        payload = wandb_log.backfill_payload(RUN_OK, ARTIFACTS)
        wb = FakeWandbModule()
        wandb_log.log_backfill(payload, wandb_module=wb, new_run_id="validation-20261108T140200Z")
        self.assertEqual((wb.init_kwargs["id"], wb.init_kwargs["resume"]),
                         ("validation-20261108T140200Z", "never"))
        self.assertEqual(wb.run.artifacts[0].name, "validation-validation-20261108T140200Z")
        wb2 = FakeWandbModule()
        wandb_log.log_backfill(payload, wandb_module=wb2)
        self.assertEqual((wb2.init_kwargs["id"], wb2.init_kwargs["resume"]),
                         ("h200-20261106T185236Z", "allow"))
        self.assertEqual(wb2.run.artifacts[0].name, "h200-20261106T185236Z")
        self.assertEqual(wb.run.logged, wb2.run.logged)
        self.assertEqual(wb.run.summary, wb2.run.summary)
        self.assertEqual(wb.run.definitions, wb2.run.definitions)
        self.assertEqual(wb.run.finished, wb2.run.finished)


class WeaveInputsTests(unittest.TestCase):
    def test_rollout_inputs(self):
        with scratch() as tmp:
            task = make_task(tmp, "triton")
            from utils.config import RolloutConfig
            inputs = weave_trace.rollout_inputs(task, RolloutConfig.preset("grpo"), 1)
            self.assertEqual(inputs["backend"], "triton")
            self.assertEqual(inputs["model_new_sha256"],
                             hashlib.sha256((task / "model_new.py").read_bytes()).hexdigest())
            self.assertIn("kernels/triton/axpby.py", inputs["kernel_files"])
            self.assertEqual(inputs["turns_used"], 1)


class CLITests(unittest.TestCase):
    def test_dry_run_single_is_one_json_doc(self):
        proc = subprocess.run(
            [sys.executable, "-S", "-m", "tracking.wandb_log", "backfill",
             "--run-id", RUN_OK, "--dry-run"], cwd=ROOT,
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["run_id"], RUN_OK)

    def test_no_wandb_weave_torch_imported(self):
        code = ("import tracking.wandb_log, tracking.weave_trace; import sys; "
                "assert 'torch' not in sys.modules and 'wandb' not in sys.modules "
                "and 'weave' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


class SweepTests(unittest.TestCase):
    def test_sweep_config_matches_rollout_config(self):
        import dataclasses
        from utils.config import RolloutConfig
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not installed")
        sweep = yaml.safe_load((ROOT / "sweeps" / "rollout_sweep.yaml").read_text())
        fields = {f.name for f in dataclasses.fields(RolloutConfig)}
        self.assertTrue(set(sweep["parameters"]) <= fields, set(sweep["parameters"]) - fields)
        self.assertEqual(sweep["command"][-1], "${args_no_boolean_flags}")
        self.assertIn("scripts/sweep_rollout.sh", sweep["command"])
        self.assertEqual(sweep["metric"], {"name": "reward_mean", "goal": "maximize"})

    def test_sweep_wrapper_requires_workdir_and_gpu(self):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / "sweep_rollout.sh")], check=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("SWEEP_WORKDIR", "CUDA_VISIBLE_DEVICES")}
        proc = subprocess.run(["bash", str(ROOT / "scripts" / "sweep_rollout.sh")],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SWEEP_WORKDIR", proc.stderr)


class ContractTests(unittest.TestCase):
    def test_pipeline_keys_have_exactly_one_adr_category(self):
        import ast
        from tracking.contract import classify, rules, CATEGORIES
        from utils.config import RolloutConfig
        from utils.rollout_env import run_task
        from utils.runtime import check
        from utils.stats import summarize_bench
        from utils.provenance import summarize_events
        for r in rules():
            self.assertIn(r['category'], CATEGORIES)
            self.assertTrue(all(r[k] for k in ('namespace', 'type', 'unit', 'null', 'role', 'owner')))
        for k in RolloutConfig().to_dict(): self.assertEqual(classify('cfg', 'cfg/'+k), 'config')
        for k in run_task('/no/such/contract/task', RolloutConfig()): classify('result', k)
        for k in check(run=lambda _:None, which=lambda _:None, pkg=lambda _:None):
            self.assertEqual(classify('runtime', k), 'config')
        profile = summarize_events([('axpby_kernel', 2)], ['axpby_kernel'], 'triton')
        profile.update(timing_measured=True, t_eager_us=3, t_compile_us=1)
        for k in profile: self.assertEqual(classify('profile', k), 'table')
        samples = {impl: {'event_us':[1,2,3], 'wall_us':[2,3,4], 'graph_us':[1,1,1],
                         'prepare_s':1, 'warmup_iters':2, 'graph_capturable':True,
                         'graph_replay_ok':True, 'graph_error':None}
                   for impl in ('eager','compile','generated')}
        for k in summarize_bench(samples, 3): self.assertEqual(classify('result', k), 'table')
        # Detect metadata keys added to the benchmark producer, not just fixtures.
        tree=ast.parse((ROOT/'utils/benchmark.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value,str) and key.value.startswith('bench/'):
                        classify('result',key.value)
        from utils.grpo import build_batch
        _, stats=build_batch([],[],RolloutConfig())
        for k in stats:self.assertEqual(classify('batch',k),'metric')
        with self.assertRaises(ValueError):classify('result','env/unreviewed_new_key')
        # All real saved names, including unavailable/failed backend records.
        def leaves(value,prefix=''):
            if isinstance(value,dict):
                for k,v in value.items():yield from leaves(v,prefix+'/'+k if prefix else k)
            else:yield prefix
        for file in ARTIFACTS.glob('kernel_names-*.json'):
            for k in leaves(json.loads(file.read_text())):classify('names',k)

    def test_full_config_shares_rules_and_group_stats_backfill(self):
        from utils.config import RolloutConfig
        from utils.provenance import PROVENANCE_RULESET
        from unittest.mock import patch
        cfg=RolloutConfig.preset('cispo').to_dict()
        results={'config':cfg,'provenance_rules':PROVENANCE_RULESET,
                 'rollouts':{'one':{'reward/value':2,'reward/reason':'correct','env/provenance':1}},
                 'profiles':{'one':{'timing_measured':True,**{'provenance/'+b: (1 if b=='agent' else 0) for b in wandb_log.SHARES}}},
                 'measured_groups':{'cispo':{'stats':{k:0.0 for k in wandb_log.BATCH_KEYS}}}}
        def read(_,name,__):return results if name=='results' else {}
        with patch.object(wandb_log,'_read',side_effect=read):
            p=wandb_log.backfill_payload(RUN_OK,ARTIFACTS)
        self.assertEqual({k[4:]:v for k,v in p['config'].items() if k.startswith('cfg/')},cfg)
        self.assertEqual(p['rollouts'][0]['env/provenance_agent'],1)
        self.assertEqual(p['config']['meta/provenance_rules'],PROVENANCE_RULESET)
        self.assertIn('batch/cispo/adv/mean',p['metrics'])
        self.assertIn('batch/cispo/turns/p95',p['metrics'])
        wb=FakeWandbModule();wb.run.summary.update({'case':'hack','env/provenance':0,'reward/value':-1})
        wandb_log.log_backfill(p,wandb_module=wb)
        self.assertNotIn('case',wb.run.summary)
        self.assertFalse(any(k.startswith(('env/','meta/')) for k in wb.run.summary))
        self.assertNotIn('reward/value',wb.run.summary)
        self.assertNotIn('case',wb.run.logged[0][0])
        self.assertIn(('*',{'step_metric':'iteration','summary':'none'}),wb.run.definitions)
        self.assertIn('rollouts',wb.run.logged[0][0])
        # Header config is real recorder output; stages are explicitly synthetic.
        from tests.test_record_kernel_names import run_main, PROFILE
        from tracking.contract import classify
        with scratch() as tmp, contextlib.redirect_stdout(io.StringIO()):
            record=run_main(PROFILE,'--device','cuda','--backends','triton','--output',str(Path(tmp)/'names.json'))
        self.assertEqual(set(record['config']),set(cfg))
        self.assertEqual(record['case_configs']['triton']['backend'],'triton')
        self.assertEqual(record['provenance_rules'],PROVENANCE_RULESET)
        def leaves(value,prefix=''):
            if isinstance(value,dict):
                for k,v in value.items():yield from leaves(v,prefix+'/'+k if prefix else k)
            else:yield prefix
        for key in leaves(record):classify('names',key)
        from tracking.contract import classify
        for key in wb.run.logged[0][0]: classify('metric',key)

    def test_old_config_rules_and_shares_are_unknown_not_reconstructed(self):
        p=wandb_log.backfill_payload(RUN_OK,ARTIFACTS)
        self.assertFalse(p['config']['tracking/config_available'])
        self.assertIsNone(p['config']['meta/provenance_rules'])
        self.assertFalse(p['config']['tracking/shares_available'])
        self.assertTrue(all(r['env/provenance_agent'] is None for r in p['rollouts']))
        self.assertIn('batch/grpo/adv/mean',p['metrics'])
        self.assertIn('batch/dapo/turns/mean',p['metrics'])

    def test_live_batch_stats_and_standalone_identity(self):
        import dataclasses
        from utils.config import RolloutConfig
        from tests.support import mock_rollout
        cfg=dataclasses.replace(RolloutConfig.preset('cispo'),group_size=3)
        rows=[mock_rollout(2),mock_rollout(2),mock_rollout(-1,'provenance')]
        wb=FakeWandbModule()
        run,_,stats=wandb_log.log_batch(['task']*3,rows,cfg,step=7,environment={},wandb_module=wb)
        self.assertEqual(wb.init_kwargs['group'],'cispo')
        self.assertEqual(run.logged[0][0]['iteration'],7)
        self.assertTrue(set(stats)<=set(run.logged[0][0]))
        self.assertEqual(run.summary['cases'],3)
        self.assertNotIn('reward/value',run.summary)
        with scratch() as tmp:
            task=make_task(tmp,'triton')
            wb=FakeWandbModule()
            result={'reward/value':-1,'reward/reason':'provenance','log/stderr_tail':'trace-only'}
            profile={'provenance/agent':0,'timing_measured':True}
            run=wandb_log.log_rollout(result,cfg,workdir=task,profile=profile,environment={},wandb_module=wb)
            table=run.logged[0][0]['rollouts'];record=dict(zip(table[1],table[2][0]))
            self.assertEqual(record['meta/model_new_sha256'],hashlib.sha256((task/'model_new.py').read_bytes()).hexdigest())
            self.assertEqual(record['env/provenance_agent'],0)
            self.assertNotIn('log/stderr_tail',record)

    def test_profile_sink_does_not_change_result_or_reward(self):
        from unittest.mock import patch
        from utils.config import RolloutConfig
        from utils.rollout_env import run_task
        from tests.test_rollout_env import RolloutProtocolTests
        test=RolloutProtocolTests(); stages=test.stages()
        profile=json.loads(stages[-1][1].splitlines()[-1]);profile['provenance/agent']=1.0
        stages[-1]=(0,json.dumps(profile),'',False)
        with scratch() as tmp:
            task=make_task(tmp)
            with patch('utils.rollout_env._stage',side_effect=stages):a=run_task(task,RolloutConfig())
            sink={'old':1}
            with patch('utils.rollout_env._stage',side_effect=stages):b=run_task(task,RolloutConfig(),profile_out=sink)
        self.assertEqual(set(a),set(b));self.assertEqual(a['reward/value'],b['reward/value'])
        self.assertEqual(len(a),18);self.assertEqual(sink,profile)

    def test_validate_persists_actual_case_and_group_configs(self):
        from unittest.mock import patch
        from scripts import validate
        from utils.config import RolloutConfig
        from utils.rollout_env import run_task
        base=run_task('/no/such/task',RolloutConfig())
        def evaluate(path,cfg,turns,*,profile_out):
            profile_out.update({'timing_measured':False,'t_gen_us':None,
                                **{'provenance/'+b:None for b in wandb_log.SHARES}})
            return dict(base,**{'meta/algo':cfg.algo,'env/turns_used':turns})
        with scratch() as tmp:
            root=Path(tmp);(root/'artifacts').mkdir()
            with patch.object(validate,'ROOT',root),patch.object(validate,'run_task',side_effect=evaluate),patch.object(sys,'argv',['validate','--device','cpu','--timeout-s','17']),patch.dict(os.environ),contextlib.redirect_stdout(io.StringIO()):
                validate.main()
            record=json.loads((root/'artifacts/results.json').read_text())
        self.assertEqual(record['config']['timeout_s'],17)
        self.assertEqual(record['case_configs']['timeout']['timeout_s'],5)
        self.assertEqual(record['case_configs']['overlong']['algo'],'dapo')
        self.assertEqual(record['group_configs']['cispo']['group_size'],3)
        self.assertEqual(record['group_configs']['cispo']['algo'],'cispo')
        self.assertTrue(all(len(v)==64 for v in record['candidate_sha256'].values()))
        self.assertEqual(set(record['profiles']),set(record['rollouts']))
        self.assertTrue(record['provenance_rules'])
        from tracking.contract import classify
        for source in ('config','case_configs','group_configs'):
            for k in record[source]:
                if source=='config':classify('results',source+'/'+k)
                else:
                    for field in record[source][k]:classify('results',source+'/'+k+'/'+field)

    def test_weave_retains_raw_profile_and_stage_tails_without_sdks(self):
        from unittest.mock import patch
        from utils.config import RolloutConfig
        from tracking import weave_trace
        cfg=RolloutConfig()
        with scratch() as tmp:
            task=make_task(tmp)
            record={'result':{'log/stderr_tail':'only trace','reward/value':-1},
                    'profile':{'provenance/agent':0}}
            sink={}
            with patch.object(weave_trace,'ops',return_value={'run_task':lambda inputs:record}):
                result=weave_trace.traced_run_task(task,cfg,profile_out=sink)
            self.assertEqual(result,record['result']);self.assertEqual(sink,record['profile'])



if __name__ == "__main__":
    unittest.main()
