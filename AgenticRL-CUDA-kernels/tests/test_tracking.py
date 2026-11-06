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
    def __init__(self):
        self.logged, self.artifacts, self.summary = [], [], {}
        self.url, self.id, self.finished = "https://fake/run", "rid", None

    def log(self, row, step=None):
        self.logged.append((row, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.finished = exit_code


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path):
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
                  "artifact/trace": "/p", "bench/available": False}
        self.assertEqual(wandb_log.metrics_for_wandb(result),
                         {"env/verify_ok": True, "reward/value": 2.0,
                          "bench/available": False})
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
        self.assertEqual(len(wandb.run.logged), 13)   # 12 cases + 1 tables call
        self.assertEqual(wandb.run.finished, 0)
        wandb2 = FakeWandbModule()
        wandb_log.log_backfill(wandb_log.backfill_payload(RUN_FAILED, ARTIFACTS),
                               wandb_module=wandb2)
        self.assertEqual(wandb2.run.finished, 1)


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
        self.assertEqual(sweep["metric"], {"name": "reward/value", "goal": "maximize"})

    def test_sweep_wrapper_requires_workdir_and_gpu(self):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / "sweep_rollout.sh")], check=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("SWEEP_WORKDIR", "CUDA_VISIBLE_DEVICES")}
        proc = subprocess.run(["bash", str(ROOT / "scripts" / "sweep_rollout.sh")],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SWEEP_WORKDIR", proc.stderr)


if __name__ == "__main__":
    unittest.main()
