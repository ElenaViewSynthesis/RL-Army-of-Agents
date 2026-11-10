import copy
import json
import types
import unittest
from pathlib import Path

from scripts.smoke_tracking_live import prepare, review_wandb, review_weave
from tests.support import ROOT, scratch


class PreparationTests(unittest.TestCase):
    def test_partial_source_is_frozen_and_not_promoted_to_new_gpu_evidence(self):
        with scratch() as tmp:
            output = Path(tmp) / "payload"
            payload = prepare("20261108T014026Z", ROOT / "artifacts", output)
            self.assertIn("source-partial_acceptance_invalid_baseline", payload["tags"])
            self.assertIn("Not a new GPU measurement", payload["notes"])
            self.assertEqual(payload, json.loads((output / "payload.json").read_text()))
            source = json.loads((ROOT / "artifacts/results-20261108T014026Z.json").read_text())
            self.assertEqual(payload["metrics_by_case"], source["rollouts"])
            self.assertTrue(all(Path(path).is_file() for path in payload["files"]))
            self.assertTrue(all(Path(path).parent == output / "source" for path in payload["files"]))
            with self.assertRaises(FileExistsError):
                prepare("20261108T014026Z", ROOT / "artifacts", output)


class ReadbackTests(unittest.TestCase):
    def payload(self):
        return {"run_id": "source", "config": {"cfg/iters": 10},
                "summary": {"cases": 2, "reward_mean": 0.5},
                "metrics": {"batch/grpo/reward/mean": 0.5},
                "rollouts": [{"case": "honest", "extra": 1}, {"case": "hack"}],
                "kernels": [{"backend": "triton", "event": "authored"}], "files": [],
                "metrics_by_case": {
                    "honest": {"meta/backend": "triton", "reward/value": 2},
                    "hack": {"meta/backend": "triton", "reward/value": -1}}}

    def test_weave_query_and_exact_results(self):
        payload = self.payload()
        calls = [types.SimpleNamespace(
            id=case, inputs={"case": case, "run_id": "source", "backend": "triton"},
            exception=None, output={**result, "backfilled": True, "source": "artifacts"})
            for case, result in payload["metrics_by_case"].items()]
        from unittest.mock import Mock
        client = Mock()
        client.get_calls.return_value = calls
        review = review_weave(client, "validation", payload)
        self.assertEqual(review["status"], "passed")
        client.get_calls.assert_called_once_with(
            query={"$expr": {"$eq": [{"$getField": "attributes.tracking_validation_id"},
                                      {"$literal": "validation"}]}}, limit=3)
        calls[1].output["reward/value"] = 2
        with self.assertRaisesRegex(AssertionError, "raw result differs"):
            review_weave(client, "validation", payload)
        client.get_calls.return_value = calls[:1]
        with self.assertRaisesRegex(AssertionError, "count differs"):
            review_weave(client, "validation", payload)

    def test_wandb_tables_snapshot_and_summary_are_checked(self):
        payload = self.payload()
        with scratch() as tmp:
            output = Path(tmp)
            summary = dict(payload["summary"])
            def file(name):
                def download(root, exist_ok):
                    target = Path(root)
                    target.mkdir()
                    columns = list(dict.fromkeys(key for row in payload[name] for key in row))
                    path = target / "table.json"
                    path.write_text(json.dumps({"columns": columns, "data": [
                        [row.get(key) for key in columns] for row in payload[name]]}))
                    return types.SimpleNamespace(name=str(path))
                return types.SimpleNamespace(download=download)
            raw = output / "raw"
            raw.mkdir()
            artifact = types.SimpleNamespace(type="validation-run", download=lambda root: str(raw))
            remote = types.SimpleNamespace(
                state="finished", config=payload["config"], summary=summary, file=file,
                scan_history=lambda: [{"iteration": 0, **payload["metrics"],
                                       "rollouts": {"path": "rollouts"}, "kernels": {"path": "kernels"}}],
                logged_artifacts=lambda: [artifact], url="test-url")
            api = types.SimpleNamespace(run=lambda path: remote)
            review = review_wandb(api, "entity/project/validation", payload, output)
            self.assertEqual(review["status"], "passed")
            self.assertEqual(review["tables"]["rollouts"][1]["extra"], None)
            remote.summary = dict(summary, **{"reward/value": -1})
            with self.assertRaisesRegex(AssertionError, "summary fields"):
                review_wandb(api, "entity/project/validation", payload, output)
            remote.summary = copy.deepcopy(summary)
            remote.config = {"cfg/iters": 20}
            with self.assertRaisesRegex(AssertionError, "config differs"):
                review_wandb(api, "entity/project/validation", payload, output)


if __name__ == "__main__":
    unittest.main()
