import csv
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

# Kept literal (not imported from utils.diagnose) so review stays Torch-free and cheap.
NVTX_RANGES = ("agenticrl_rollout", "agenticrl_warmup", "agenticrl_eager",
               "agenticrl_compile", "agenticrl_generated")
CAPTURE_RANGES = ("agenticrl_generated", "agenticrl_rollout")
# Nsight Compute raw-page metric -> (summary field, required unit or None).
NCU_FIELDS = (
    ("gpu__time_duration.sum", "duration_us", "us"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_throughput_pct", "%"),
    # Name confirmed in the H200 basic-set export (ncu 2026.2.1, tests/fixtures/diagnostics/ncu-cuda-clock-none).
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram_throughput_pct", "%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved_occupancy_pct", "%"),
    ("launch__registers_per_thread", "registers_per_thread", None),
    ("launch__shared_mem_per_block_static", "shared_mem_static_bytes", "byte/block"),
    ("launch__shared_mem_per_block_dynamic", "shared_mem_dynamic_bytes", "byte/block"),
    ("launch__grid_size", "grid_size", None),
    ("launch__block_size", "block_size", None),
)
NCU_SUMMARY_FIELDS = tuple(dict.fromkeys(field for _, field, _ in NCU_FIELDS))
# Metrics a preset must yield, or the capture is rejected (fail closed).
NCU_REQUIRED = {
    "basic": (("gpu__time_duration.sum", "us"), ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%")),
    "roofline": (("gpu__time_duration.sum", "us"), ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%")),
    "launch": (("launch__grid_size", None), ("launch__block_size", None)),
}
# nsys GPU-metrics names (TARGET_INFO_GPU_METRICS.metricName) read for the summary.
SM_ACTIVE_PREFIX = "SM Active"
DRAM_PREFIXES = ("DRAM Read Bandwidth", "DRAM Write Bandwidth")


def range_label(name):
    return name[len("agenticrl_"):] if name.startswith("agenticrl_") else name


def range_summary(ranges, kernels, agent_names, backend):
    """Per NVTX range: kernel count, union busy time and time by provenance class.

    ranges: {name: (start_ns, end_ns)}; kernels: (start_ns, end_ns, name).
    A kernel belongs to a range when it starts and ends inside it. Classes come
    from utils.provenance.classify_kernel by name; with no recorded authored
    names (older reports) the class times are None.
    """
    from utils.provenance import classify_kernel
    summary = {}
    for name, (start, end) in ranges.items():
        inside = [(a, b, kernel) for a, b, kernel in kernels if a >= start and b <= end]
        classes = None
        if agent_names is not None and backend is not None:
            classes = {key: 0 for key in ("agent", "library", "torch", "memory")}
            for a, b, kernel in inside:
                classes[classify_kernel(kernel, agent_names, backend)] += b - a
        summary[name] = {"kernel_count": len(inside),
                         "busy_ns": union_duration([(a, b) for a, b, _ in inside]),
                         **{f"{key}_ns": (classes[key] if classes is not None else None)
                            for key in ("agent", "library", "torch", "memory")}}
    return summary


def gpu_metrics_means(db, start, end):
    """Mean SM-active and DRAM-throughput percentages inside [start, end].

    Reads GPU_METRICS(timestamp, metricId, value) joined to
    TARGET_INFO_GPU_METRICS(metricId, metricName). DRAM throughput is read plus
    write bandwidth percentage at each sample. Returns (None, None, reason)
    when the tables or metrics are absent.
    """
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        return None, None, "gpu_metrics_tables_absent"
    names = dict(db.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS").fetchall())
    rows = db.execute("SELECT timestamp, metricId, value FROM GPU_METRICS WHERE timestamp>=? AND timestamp<=?",
                      (start, end)).fetchall()
    sm, dram = [], {}
    for timestamp, metric, value in rows:
        label = str(names.get(metric, ""))
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if label.startswith(SM_ACTIVE_PREFIX):
            sm.append(float(value))
        elif label.startswith(DRAM_PREFIXES):
            dram[timestamp] = dram.get(timestamp, 0.0) + float(value)
    sm_mean = sum(sm) / len(sm) if sm else None
    dram_mean = sum(dram.values()) / len(dram) if dram else None
    reason = None if sm_mean is not None and dram_mean is not None else "gpu_metrics_samples_absent_in_range"
    return sm_mean, dram_mean, reason


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
        summary = ("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                   if result["diag/check"] == "racecheck" else "========= ERROR SUMMARY: 0 errors")
        accepted = {"========= COMPUTE-SANITIZER", summary}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines += [line.strip() for line in tool_output(directory, review).splitlines()
                  if line.lstrip().startswith("=========")]
        if accepted <= set(lines) and all(line in accepted for line in lines):
            set_status(review, "passed", "sanitizer_clean")
        else:
            set_status(review, "unavailable", "unrecognized_sanitizer_output")


def tool_output(directory, review):
    text = []
    for name in ("tool.stdout.log", "tool.stderr.log"):
        path = directory / name
        if path.is_file():
            text.append(path.read_text(errors="replace"))
            review["raw_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return "\n".join(text)


def capture_worker_matches(captured):
    worker = captured.get("worker") or {}
    return (worker_matches(captured, "capture") and worker.get("captured_forwards") == 1
            and worker.get("verification_checks") == 2
            and worker.get("capture_range") == "agenticrl_generated"
            and isinstance(worker.get("warmup_iters"), int) and worker["warmup_iters"] > 0)


def union_duration(intervals):
    total, last_end = 0, None
    for start, end in sorted(intervals):
        if last_end is None or end > last_end:
            total += end - (start if last_end is None else max(start, last_end))
            last_end = end
    return total


def review_nsys(captured, directory, review):
    text = tool_output(directory, review)
    sqlite_path, report_path = directory / "capture.sqlite", directory / "capture.nsys-rep"
    recovered = directory / "postprocess.json"
    if recovered.is_file():
        receipt = json.loads(recovered.read_text())
        supplement = directory / "postprocess-v2.json"
        if supplement.is_file():
            updated = json.loads(supplement.read_text())
            if updated.get("base_receipt_sha256") != hashlib.sha256(recovered.read_bytes()).hexdigest():
                set_status(review, "failed", "postprocess_provenance_mismatch")
                return
            receipt = updated
        source_qdstrm = directory / ("original.qdstrm" if "processed_qdstrm_sha256" in receipt
                                     else "capture.qdstrm")
        required = {"capture.sqlite": sqlite_path, "capture.nsys-rep": report_path}
        if (receipt.get("complete") is not True or receipt.get("gpu_recaptured") is not False
                or receipt.get("source_report_sha256") != review["capture_sha256"]
                or not source_qdstrm.is_file()
                or receipt.get("source_qdstrm_sha256") != hashlib.sha256(
                    source_qdstrm.read_bytes()).hexdigest()
                or "processed_qdstrm_sha256" in receipt and receipt["processed_qdstrm_sha256"]
                   != hashlib.sha256((directory / "capture.qdstrm").read_bytes()).hexdigest()
                or any(not path.is_file() or receipt.get("output_sha256", {}).get(name)
                       != hashlib.sha256(path.read_bytes()).hexdigest() for name, path in required.items())
                or len(receipt.get("stages", [])) != 2
                or any(stage.get("returncode") != 0 or stage.get("timed_out")
                       for stage in receipt.get("stages", []))):
            set_status(review, "failed", "postprocess_provenance_mismatch")
            return
        review["raw_sha256"]["postprocess.json"] = hashlib.sha256(recovered.read_bytes()).hexdigest()
        review["observations"]["recovery_run_id"] = receipt.get("recovery_run_id")
    elif "Unable to retrieve the importer version: skipping importation of the QDSTRM file." in text:
        review["observations"]["raw_qdstrm_saved"] = (directory / "capture.qdstrm").is_file()
        set_status(review, "unavailable", "nsys_importer_unavailable")
        return
    if not sqlite_path.is_file() or not report_path.is_file():
        set_status(review, "failed", captured["result"]["diag/reason"])
        return
    tool = next(stage for stage in captured["stages"] if stage["label"] == "tool")
    capture_flag = next((arg for arg in tool["argv"] if arg.startswith("--nvtx-capture=")), "")
    capture_range = capture_flag[len("--nvtx-capture="):]
    if (tool["returncode"] != 0 or not capture_worker_matches(captured)
            or "--capture-range=nvtx" not in tool["argv"]
            or capture_range not in CAPTURE_RANGES
            or "--discard-environment=true" not in tool["argv"]):
        set_status(review, "failed", "timeline_capture_evidence_incomplete")
        return
    gpu_metrics_requested = any(arg.startswith("--gpu-metrics-devices=") for arg in tool["argv"])
    try:
        with sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            named = db.execute(
                "SELECT COALESCE(n.text,s.value),n.start,n.end FROM NVTX_EVENTS n "
                "LEFT JOIN StringIds s ON n.textId=s.id").fetchall()
            all_kernels = db.execute(
                "SELECT k.start,k.end,k.deviceId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "LEFT JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start,k.end").fetchall()
            found = {}
            for name, a, b in named:
                if name in NVTX_RANGES:
                    found.setdefault(name, []).append((a, b))
            if (len(found.get("agenticrl_generated", [])) != 1
                    or len(found.get(capture_range, [])) != 1
                    or any(len(spans) != 1 for spans in found.values()) or not all_kernels):
                set_status(review, "failed", "empty_or_ambiguous_timeline")
                return
            ranges = {name: spans[0] for name, spans in found.items()}
            outer_start, outer_end = ranges[capture_range]
            start, end = ranges["agenticrl_generated"]
            if (any(not isinstance(a, int) or not isinstance(b, int) or b <= a
                    for a, b in ranges.values())
                    or not outer_start <= start < end <= outer_end
                    or any(not isinstance(a, int) or not isinstance(b, int) or b <= a
                           or a < outer_start or b > outer_end or device != captured["worker"]["gpu_index"]
                           or not isinstance(name, str)
                           for a, b, device, name in all_kernels)):
                set_status(review, "failed", "invalid_or_out_of_range_kernel_events")
                return
            kernels = [kernel for kernel in all_kernels if kernel[0] >= start and kernel[1] <= end]
            if not kernels:
                set_status(review, "failed", "empty_or_ambiguous_timeline")
                return
            api = db.execute(
                "SELECT s.value,r.start,r.end,r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "LEFT JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.end<=? ORDER BY r.start",
                (start, end)).fetchall()
            memcpy_present = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'").fetchone()
            metrics = gpu_metrics_means(db, start, end) if gpu_metrics_requested else None
    except sqlite3.Error as exc:
        review["observations"]["sqlite_error_type"] = type(exc).__name__
        set_status(review, "failed", "unsupported_timeline_schema")
        return
    agent_names = captured.get("agent_kernel_names")
    per_range = range_summary(ranges, [(a, b, name) for a, b, _, name in all_kernels],
                              agent_names, captured["result"].get("diag/backend"))
    review["observations"]["nvtx_ranges"] = {name: {"start_ns": a, "end_ns": b}
                                             for name, (a, b) in ranges.items()}
    review["observations"]["range_summary"] = per_range
    for name, values in per_range.items():
        for key, value in values.items():
            review["result"][f"diag/nsys/{range_label(name)}/{key}"] = value
    if metrics is not None:
        sm_mean, dram_mean, reason = metrics
        review["result"].update({"diag/nsys/generated/sm_active_pct_mean": sm_mean,
                                 "diag/nsys/generated/dram_throughput_pct_mean": dram_mean,
                                 "diag/nsys/gpu_metrics_reason": reason})
    busy = union_duration([(a, b) for a, b, _, _ in kernels])
    span = max(b for _, b, _, _ in kernels) - min(a for a, _, _, _ in kernels)
    review["raw_sha256"]["capture.sqlite"] = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    review["observations"].update(
        range_start_ns=start, range_end_ns=end,
        kernels=[{"start_ns": a, "end_ns": b, "device_index": device, "name": name}
                 for a, b, device, name in kernels],
        runtime_api=[{"name": name, "start_ns": a, "end_ns": b, "return_value": value}
                     for name, a, b, value in api],
        transfer_duration_ns=None,
        transfer_reason=("Memcpy activity exists but is not decoded by this parser." if memcpy_present
                         else "No memcpy activity table was present in these captured fixtures."))
    review["result"].update({"diag/kernel_count": len(kernels), "diag/nsys_kernel_busy_ns": busy,
                             "diag/nsys_kernel_span_ns": span, "diag/nsys_kernel_gap_ns": span - busy,
                             "diag/nsys_range_ns": end - start})
    review["limitations"].append(
        "Kernel gaps cover only the captured process/range, not whole-device idle time or utilization.")
    set_status(review, "passed", "generated_forward_timeline_captured")


def review_ncu(captured, directory, review):
    text = tool_output(directory, review)
    if "Failed to lock GPU clock frequencies!" in text:
        set_status(review, "unavailable", "gpu_clock_control_unavailable")
    elif "ERR_NVGPUCTRPERM" in text:
        set_status(review, "unavailable", "gpu_counter_permission")
        return
    if review["result"]["diag/status"] == "unavailable":
        return
    path = directory / "metrics.stdout.log"
    if not path.is_file():
        set_status(review, "failed", captured["result"]["diag/reason"])
        return
    stages = {stage["label"]: stage for stage in captured["stages"]}
    tool, metrics = stages.get("tool", {}), stages.get("metrics", {})
    argv = tool.get("argv", [])
    selector = review["result"].get("diag/kernel_selector")
    # Reports from before presets existed ran --set basic.
    preset = review["result"].get("diag/ncu_preset") or "basic"
    if (tool.get("returncode") != 0 or metrics.get("returncode") != 0
            or argument(argv, "--clock-control") != "none"
            or argument(argv, "--launch-count") != "1"
            or argument(argv, "--nvtx-include") != "agenticrl_generated/"
            or not selector or argument(argv, "--kernel-name") != "regex:" + selector
            or not capture_worker_matches(captured)):
        set_status(review, "failed", "counter_capture_evidence_incomplete")
        return
    with path.open(newline="") as stream:
        table = list(csv.reader(stream))
    review["raw_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(table) != 3:
        set_status(review, "failed", "unsupported_counter_csv")
        return
    header, units, values = table
    if (len(set(header)) != len(header) or len(header) != len(units) or len(header) != len(values)
            or not {"ID", "Kernel Name", "Device", "CC"} <= set(header)):
        set_status(review, "failed", "unsupported_counter_csv")
        return
    split = header.index("CC") + 1
    if not {"ID", "Kernel Name", "Device", "CC"} <= set(header[:split]):
        set_status(review, "failed", "unsupported_counter_csv")
        return
    metadata = dict(zip(header[:split], values[:split]))
    worker = captured["worker"]
    if (any(units[:split]) or not metadata["ID"].isdigit()
            or metadata["Device"] != str(worker.get("gpu_index"))
            or metadata["CC"] != worker.get("compute_capability")
            or re.search(selector, metadata["Kernel Name"]) is None):
        set_status(review, "failed", "counter_identity_or_selector_mismatch")
        return
    fields = {name: {"value": value if value else None, "unit": unit if unit else None}
              for name, value, unit in zip(header[split:], values[split:], units[split:])}
    if preset not in NCU_REQUIRED:
        set_status(review, "failed", "unknown_ncu_preset")
        return
    for name, unit in NCU_REQUIRED[preset]:
        metric = fields.get(name, {})
        value = number(metric.get("value"))
        if (value is None or value < 0 or (unit is not None and metric.get("unit") != unit)
                or name == "gpu__time_duration.sum" and value <= 0):
            set_status(review, "failed", "required_counter_metric_unavailable")
            return
    label = kernel_label(metadata["Kernel Name"], captured.get("agent_kernel_names"))
    summary = ncu_summary(fields)
    review["observations"]["profiles"] = [{"metadata": metadata, "metrics": fields}]
    review["observations"]["ncu_summary"] = {label: summary}
    review["result"].update({"diag/kernel_count": 1, "diag/ncu_metric_fields": len(fields),
                             "diag/selector_matched": True, "diag/ncu_kernel": label,
                             **{f"diag/ncu/{label}/{key}": value for key, value in summary.items()}})
    write_ncu_markdown(directory / "ncu-summary.md", label, metadata, summary, preset)
    review["result"]["diag/ncu_summary_md"] = str(directory / "ncu-summary.md")
    set_status(review, "passed", "selected_kernel_counters_captured")


def number(value):
    try:
        result = float(str(value).replace(",", "")) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def kernel_label(name, agent_names):
    """The authored function name found in the profiled kernel's name, else its base name."""
    for agent in sorted(agent_names or (), key=len, reverse=True):
        if agent and re.search(r"(?<![A-Za-z0-9_])" + re.escape(agent) + r"(?![A-Za-z0-9])", name):
            return agent
    base = re.split(r"[(<]", name, maxsplit=1)[0].strip().split("::")[-1].split()[-1:] or ["kernel"]
    return re.sub(r"[^A-Za-z0-9_]", "_", base[0]) or "kernel"


def ncu_summary(fields):
    """One kernel's summary values from the raw page; unparseable or wrong-unit values are null."""
    summary = {key: None for key in NCU_SUMMARY_FIELDS}
    for metric, key, unit in NCU_FIELDS:
        entry = fields.get(metric)
        if entry is None or summary[key] is not None:
            continue
        if unit is not None and entry.get("unit") != unit:
            continue
        value = number(entry.get("value"))
        if value is None or value < 0:
            continue
        summary[key] = int(value) if key in ("registers_per_thread", "grid_size", "block_size",
                                             "shared_mem_static_bytes", "shared_mem_dynamic_bytes") \
            and value.is_integer() else value
    return summary


def write_ncu_markdown(path, label, metadata, summary, preset):
    rows = "\n".join(f"| `{key}` | {'null' if value is None else value} |" for key, value in summary.items())
    path.write_text(
        f"# Nsight Compute summary: `{label}`\n\n"
        f"Preset `{preset}`; one launch (`--launch-count 1`), clocks not locked "
        f"(`--clock-control none`). Diagnostic only; never a reward input.\n\n"
        f"Kernel: `{metadata['Kernel Name']}` (device {metadata['Device']}, CC {metadata['CC']}).\n\n"
        f"| Field | Value |\n|---|---|\n{rows}\n\n"
        "Source: `ncu --import capture.ncu-rep --csv --page raw` (metrics.stdout.log). "
        "Null means the metric was absent from this preset or not parseable.\n")


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
    review["result"]["diag/review_json"] = str(path.parent / "reviewed.json")
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
    (path.parent / "reviewed.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return review
