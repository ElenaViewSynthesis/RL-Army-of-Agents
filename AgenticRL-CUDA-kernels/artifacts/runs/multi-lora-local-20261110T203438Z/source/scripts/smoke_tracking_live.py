import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from tracking import wandb_log


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def prepare(source_run_id, artifacts, output):
    artifacts, output = Path(artifacts), Path(output)
    required = ("results", "kernel_names", "environment", "acceptance-review")
    files = [artifacts / f"{name}-{source_run_id}.json" for name in required]
    check(all(path.is_file() for path in files), "Source evidence bundle is incomplete")
    acceptance = json.loads(files[-1].read_text())
    check(acceptance["run_id"] == source_run_id, "Acceptance run ID differs from source")
    files += [artifacts / f"gpu_tests.stderr-{source_run_id}.log",
              artifacts / f"require.stdout-{source_run_id}.log"]
    check(all(path.is_file() for path in files), "Source test/probe logs are missing")
    manifest = {"runs": [{
        "run_id": source_run_id, "kind": "tracking-validation", "status": "finished",
        "notes": f"Live service validation replay of {source_run_id}; source acceptance "
                 f"is {acceptance['status']}. Not a new GPU measurement or training iteration.",
        "files": [str(path) for path in files],
    }]}
    payload = wandb_log.backfill_payload(source_run_id, artifacts, manifest)
    payload["tags"] = ["tracking-validation", "replay", f"source-{acceptance['status']}"]
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "source"
    staged.mkdir()
    for path in files:
        shutil.copy2(path, staged / path.name)
    payload["files"] = [str((staged / path.name).resolve()) for path in files]
    (output / "payload.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def review_wandb(api, run_path, payload, output):
    remote = api.run(run_path)
    summary = dict(remote.summary)
    check(remote.state == "finished", "Live tracking validation did not finish")
    for key, value in payload["config"].items():
        check(remote.config.get(key) == value, f"Stored config differs: {key}")
    for key, value in payload["summary"].items():
        check(summary.get(key) == value, f"Stored summary differs: {key}")
    check(not (set(summary) - set(payload["summary"]) - {"rollouts", "kernels"}
               - {key for key in summary if key.startswith("_")}),
          "Per-case or unexpected summary fields were stored")
    history = list(remote.scan_history())
    check(len(history) == 1, "Validation replay must be exactly one history snapshot")
    tables = {}
    for key in ("rollouts", "kernels"):
        pointer = history[0][key]
        local = remote.file(pointer["path"]).download(root=str(output / key), exist_ok=False)
        table = json.loads(Path(local.name).read_text())
        rows = [dict(zip(table["columns"], row)) for row in table["data"]]
        columns = list(dict.fromkeys(column for row in payload[key] for column in row))
        expected_rows = [{column: row.get(column) for column in columns} for row in payload[key]]
        check(table["columns"] == columns and rows == expected_rows,
              f"Stored {key} table differs from the frozen payload")
        tables[key] = rows
    expected = {"iteration": 0, **payload["metrics"]}
    for key, value in expected.items():
        check(history[0].get(key) == value, f"Stored batch snapshot differs: {key}")
    check(not (set(history[0]) - set(expected) - {"rollouts", "kernels"}
               - {key for key in history[0] if key.startswith("_")}),
          "Per-case or unexpected history fields were stored")
    artifacts = [artifact for artifact in remote.logged_artifacts() if artifact.type == "validation-run"]
    check(len(artifacts) == 1, "Expected exactly one raw validation evidence artifact")
    downloaded = Path(artifacts[0].download(root=str(output / "raw-artifact")))
    hashes = {}
    for path in payload["files"]:
        source = Path(path)
        received = downloaded / source.name
        check(received.is_file(), f"Raw artifact file missing: {source.name}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        check(hashlib.sha256(received.read_bytes()).hexdigest() == digest,
              f"Raw artifact bytes differ: {source.name}")
        hashes[source.name] = digest
    return {"run_path": run_path, "url": remote.url, "config": remote.config,
            "summary": summary, "tables": tables, "history": history,
            "raw_artifact_sha256": hashes, "status": "passed"}


def review_weave(client, validation_id, payload):
    query = {"$expr": {"$eq": [{"$getField": "attributes.tracking_validation_id"},
                              {"$literal": validation_id}]}}
    calls = list(client.get_calls(query=query, limit=len(payload["metrics_by_case"]) + 1))
    check(len(calls) == len(payload["metrics_by_case"]), "Stored Weave replay count differs")
    found = {}
    for call in calls:
        inputs = dict(call.inputs)
        case = inputs["case"]
        check(case in payload["metrics_by_case"] and case not in found,
              "Unexpected or duplicated Weave case")
        check(inputs["run_id"] == payload["run_id"], "Weave source run ID differs")
        check(inputs["backend"] == payload["metrics_by_case"][case].get("meta/backend"),
              "Weave backend differs")
        check(call.exception is None, f"Weave replay failed: {case}")
        expected = {**payload["metrics_by_case"][case], "backfilled": True, "source": "artifacts"}
        check(dict(call.output) == expected, f"Weave raw result differs: {case}")
        found[case] = {"call_id": call.id, "inputs": inputs, "output": dict(call.output)}
    return {"query": query, "calls": found, "status": "passed"}


def finish_review(identity, payload, output, client):
    import wandb

    downloaded = output / ("readback-" + uuid4().hex[:8])
    report = {"identity": identity,
              "wandb": review_wandb(wandb.Api(timeout=30),
                                   f"{identity['entity']}/{identity['project']}/{identity['validation_id']}",
                                   payload, downloaded),
              "weave": review_weave(client, identity["validation_id"], payload),
              "ui_review": "pending", "status": "passed"}
    report["wandb_url"] = report["wandb"]["url"]
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "url": report["wandb_url"],
                      "report": str(output / "report.json")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.review_only:
        check(not args.execute, "Review-only cannot create another run")
        identity = json.loads((output / "identity.json").read_text())
        payload = json.loads((output / "payload.json").read_text())
        check(identity["source_run_id"] == payload["run_id"], "Frozen payload identity differs")
        from tracking import weave_trace
        import weave
        client = weave_trace.init(f"{identity['entity']}/{identity['project']}")
        try:
            finish_review(identity, payload, output, client)
        finally:
            weave.finish()
        return
    if not all((args.source_run_id, args.entity, args.project)):
        parser.error("--source-run-id, --entity and --project required unless --review-only")
    payload = prepare(args.source_run_id, args.artifacts.resolve(), output)
    if not args.execute:
        print(f"Dry-run payload only: {output / 'payload.json'}")
        return
    check(os.environ.get("WANDB_MODE", "online") == "online", "Online validation requires WANDB_MODE=online")
    os.environ.update(WANDB_DIR=str(output), WANDB_CACHE_DIR=str(output / "cache"),
                      WANDB_DATA_DIR=str(output / "data"), WANDB_CONSOLE="off", WANDB_DISABLE_CODE="true")
    validation_id = "tracking-check-" + uuid4().hex[:16]
    identity = {"validation_id": validation_id, "source_run_id": payload["run_id"],
                "entity": args.entity, "project": args.project, "status": "started"}
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    import weave
    from tracking import weave_trace

    client = weave_trace.init(f"{args.entity}/{args.project}")
    try:
        with weave.attributes({"tracking_validation_id": validation_id}):
            weave_trace.replay_backfill(payload)
        client.flush()
        wandb_log.log_backfill(payload, args.entity, args.project, new_run_id=validation_id)
        finish_review(identity, payload, output, client)
    finally:
        weave.finish()


if __name__ == "__main__":
    main()
