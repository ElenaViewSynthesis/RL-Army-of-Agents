import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from utils.diagnostic_review import review_report


def review_matrix(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (directory / "records.json").is_file():
        raw = json.loads((directory / "records.json").read_text())["rows"]
    else:
        raw = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines() if line]
    planned = [tuple(case) for case in manifest["planned"]]
    observed = [(row["tool"], row["backend"], row["check"], row["negative_control"]) for row in raw]
    if len(set(observed)) != len(observed) or any(case not in planned for case in observed):
        raise ValueError("Unexpected or duplicate diagnostic cases")
    rows = []
    for row in raw:
        path = (directory / row["report"]).resolve()
        if not path.is_relative_to(directory) or path.name != "report.json":
            raise ValueError("Diagnostic report must remain inside the matrix directory")
        reviewed = review_report(path)
        rows.append({key: value for key, value in row.items() if key != "result"} |
                    {"result": reviewed["result"],
                     "review": str(path.parent.relative_to(directory) / "review.json"),
                     "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    honest = [row for row in rows if not row["negative_control"]]
    controls = [row for row in rows if row["negative_control"]]
    missing = [case for case in planned if case not in observed]
    failed = [row["case"] for row in honest if row["result"]["diag/status"] in ("failed", "timeout")]
    undetected = [row["case"] for row in controls
                  if row["result"]["diag/status"] not in ("unavailable", "captured")
                  and not row["result"].get("diag/negative_control_detected", False)]
    pending_controls = [row["case"] for row in controls
                        if row["result"]["diag/status"] in ("unavailable", "captured")]
    status = "failed" if failed or undetected else (
        "pass" if not missing and not pending_controls
        and all(row["result"]["diag/status"] == "passed" for row in honest)
        else "partial")
    review = {"schema": "gpu-diagnostic-matrix-review-v1", "diagnostic_only": True,
              "run_id": manifest["run_id"], "status": status, "missing_cases": missing,
              "failed_honest_cases": failed, "undetected_controls": undetected,
              "pending_controls": pending_controls,
              "honest_status_counts": dict(Counter(row["result"]["diag/status"] for row in honest)),
              "rows": rows, "reward_inputs_changed": False,
              "limitations": ["Tool coverage is limited to the captured fixtures and runtime.",
                              "Instrumented durations and counters are not reward latencies."]}
    (directory / "review-matrix.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    review = review_matrix(args.directory)
    print(json.dumps({key: value for key, value in review.items() if key != "rows"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
