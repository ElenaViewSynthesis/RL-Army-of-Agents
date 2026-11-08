import hashlib
import json
import re
from pathlib import Path


def worker_matches(captured, mode, completed=True):
    worker, probe = captured.get("worker"), captured.get("gpu_probe")
    result = captured["result"]
    return (isinstance(worker, dict) and isinstance(probe, dict)
            and worker.get("diagnostic_only") is True and worker.get("available") is True
            and worker.get("mode") == mode
            and worker.get("backend") == result["diag/backend"]
            and worker.get("gpu_uuid") is not None
            and worker["gpu_uuid"] == probe.get("gpu_uuid") == result["diag/gpu_uuid"]
            and (not completed or worker.get("completed") is True))


def set_status(review, status, reason):
    review["result"].update({"diag/status": status, "diag/reason": reason,
                             "diag/evidence_reviewed": status != "captured"})


def argument(argv, flag):
    index = argv.index(flag) + 1 if flag in argv else len(argv)
    return argv[index] if index < len(argv) else None


def review_sanitizer(captured, directory, review):
    path = directory / "sanitizer.log"
    if not path.is_file():
        set_status(review, "failed", "missing_sanitizer_summary")
        return
    text = path.read_text(errors="replace")
    review["raw_sha256"]["sanitizer.log"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = review["result"]
    if result["diag/check"] == "racecheck":
        matches = re.findall(
            r"(?m)^========= RACECHECK SUMMARY: (\d+) hazards? displayed "
            r"\((\d+) errors?, (\d+) warnings?\)\s*$", text)
        if len(matches) != 1:
            set_status(review, "failed", "missing_or_ambiguous_sanitizer_summary")
            return
        hazards, errors, warnings = map(int, matches[0])
        result.update({"diag/reported_errors": errors, "diag/reported_warnings": warnings,
                       "diag/reported_hazards": hazards})
    else:
        matches = re.findall(r"(?m)^========= ERROR SUMMARY: (\d+) errors?\s*$", text)
        if len(matches) != 1:
            set_status(review, "failed", "missing_or_ambiguous_sanitizer_summary")
            return
        errors, warnings, hazards = int(matches[0]), 0, 0
        result["diag/reported_errors"] = errors
    stages = {stage["label"]: stage for stage in captured["stages"]}
    tool = stages["tool"]
    argv = tool["argv"]
    check = result["diag/check"]
    if argument(argv, "--tool") != check or argument(argv, "--error-exitcode") != "86":
        set_status(review, "failed", "sanitizer_command_mismatch")
        return
    if result["diag/negative_control"]:
        detected = (check == "memcheck" and errors > 0 and tool["returncode"] not in (0, None)
                    and worker_matches(captured, "invalid-access", completed=False)
                    and re.search(r"(?m)^========= Invalid __global__ write of size \d+ bytes\s*$", text)
                    is not None and "diagnostic_invalid_store" in text)
        result["diag/negative_control_detected"] = detected
        set_status(review, "failed", "invalid_access_detected" if detected else "negative_control_not_detected")
        return
    if errors or warnings or hazards:
        versions = re.findall(r"cudaVersion argument \((\d+)\) exceeds the driver version \((\d+)\)", text)
        if versions:
            review["observations"]["cuda_api_version_mismatches"] = [
                {"requested": int(requested), "driver": int(driver)}
                for requested, driver in sorted(set(versions))]
        set_status(review, "failed", "cuda_api_version_findings" if versions else "sanitizer_findings")
    elif tool["returncode"] != 0:
        set_status(review, "failed", "tool_failed")
    elif not worker_matches(captured, "verify") or captured["worker"].get("verification_checks") != 5:
        set_status(review, "failed", "verification_evidence_incomplete")
    else:
        set_status(review, "passed", "sanitizer_clean")


def tool_output(directory, review):
    text = []
    for name in ("tool.stdout.log", "tool.stderr.log"):
        path = directory / name
        if path.is_file():
            text.append(path.read_text(errors="replace"))
            review["raw_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return "\n".join(text)


def review_nsys(captured, directory, review):
    text = tool_output(directory, review)
    if "Unable to retrieve the importer version: skipping importation of the QDSTRM file." in text:
        review["observations"]["raw_qdstrm_saved"] = (directory / "capture.qdstrm").is_file()
        set_status(review, "unavailable", "nsys_importer_unavailable")
    elif (directory / "capture.sqlite").is_file():
        set_status(review, "captured", "timeline_requires_review")
    else:
        set_status(review, "failed", captured["result"]["diag/reason"])


def review_ncu(captured, directory, review):
    text = tool_output(directory, review)
    if "Failed to lock GPU clock frequencies!" in text:
        set_status(review, "unavailable", "gpu_clock_control_unavailable")
    elif "ERR_NVGPUCTRPERM" in text:
        set_status(review, "unavailable", "gpu_counter_permission")
    elif (directory / "metrics.stdout.log").is_file():
        set_status(review, "captured", "counter_metrics_require_review")
    else:
        set_status(review, "failed", captured["result"]["diag/reason"])


def review_report(path):
    path = Path(path)
    captured = json.loads(path.read_text())
    review = {"schema": "gpu-diagnostic-review-v1",
              "capture_report": str(path), "diagnostic_only": True,
              "reviewer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "result": dict(captured["result"]), "observations": {}, "raw_sha256": {},
              "limitations": ["Coverage is limited to the recorded fixtures and runtime.",
                              "Name-based selectors are not proof of authorship.",
                              "Instrumented and replayed durations never feed rewards."]}
    review["result"]["diag/review_json"] = str(path.parent / "analysis.json")
    stages = captured.get("stages", [])
    if captured.get("schema") != "gpu-diagnostics-v1" or captured.get("diagnostic_only") is not True:
        set_status(review, "failed", "capture_schema_mismatch")
    elif any(stage.get("timed_out") for stage in stages):
        set_status(review, "timeout", captured["result"]["diag/reason"])
    elif review["result"]["diag/status"] == "unavailable":
        set_status(review, "unavailable", review["result"]["diag/reason"])
    elif not any(stage.get("label") == "tool" for stage in stages):
        set_status(review, "failed", review["result"]["diag/reason"])
    elif review["result"]["diag/tool"] == "sanitizer":
        review_sanitizer(captured, path.parent, review)
    elif review["result"]["diag/tool"] == "nsys":
        review_nsys(captured, path.parent, review)
    elif review["result"]["diag/tool"] == "ncu":
        review_ncu(captured, path.parent, review)
    (path.parent / "analysis.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return review
