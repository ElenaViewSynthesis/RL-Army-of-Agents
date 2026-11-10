"""Resume polling on a detached Modal FunctionCall without resubmitting it.

`python -m cloud.modal_job --call-id fc-...` re-attaches to a call whose local
entrypoint died; it never imports the runner or allocates an app/image/GPU.
"""

import argparse
import json
import re
import sys

CALL_ID = re.compile(r"fc-[A-Za-z0-9]+")


def wait_for_call(call):
    print(json.dumps({"function_call_id": call.object_id}), flush=True)
    while True:
        try:
            return call.get(timeout=30)
        except Exception as exc:
            if type(exc) is TimeoutError:
                continue
            print(json.dumps({
                "function_call_id": call.object_id,
                "error_type": type(exc).__name__,
                "recovery": f"python -m cloud.modal_job --call-id {call.object_id}",
            }), file=sys.stderr, flush=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cloud.modal_job")
    parser.add_argument("--call-id", required=True)
    args = parser.parse_args(argv)
    if not CALL_ID.fullmatch(args.call_id):
        parser.error("--call-id must match fc-[A-Za-z0-9]+")
    import modal
    result = wait_for_call(modal.FunctionCall.from_id(args.call_id).hydrate())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
