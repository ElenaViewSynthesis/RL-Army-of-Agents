import hashlib
import json
from pathlib import Path


def review_report(path):
    path = Path(path)
    captured = json.loads(path.read_text())
    review = {"schema": "gpu-diagnostic-review-v1",
              "capture_report": str(path), "diagnostic_only": True,
              "reviewer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "result": dict(captured["result"]), "observations": {},
              "limitations": ["Raw tool evidence has not yet been interpreted."]}
    (path.parent / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return review
