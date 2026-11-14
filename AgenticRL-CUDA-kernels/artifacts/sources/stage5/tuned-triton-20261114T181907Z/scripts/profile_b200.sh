#!/usr/bin/env bash
# Profile honest task solutions on a Modal B200 with Nsight Systems / Nsight Compute.
# Runs from anywhere; always operates inside AgenticRL-CUDA-kernels/. Diagnostic only.
#
#   scripts/profile_b200.sh plan    TASKS [BACKENDS] [CASE_SET] [TOOLS]   # free: print what would run
#   scripts/profile_b200.sh launch  TASKS [BACKENDS] [CASE_SET] [TOOLS]   # PAID: start a detached B200 run
#   scripts/profile_b200.sh fetch   RUN_ID                                # download the run to artifacts/runs/RUN_ID
#   scripts/profile_b200.sh status  RUN_ID                                # rows finished so far (fetches progress)
#   scripts/profile_b200.sh kernels RUN_ID [TOP]                          # top kernels of every capture
#   scripts/profile_b200.sh query   RUN_ID QUERY_NSYS_ARGS...             # any query_nsys command on every capture
#   scripts/profile_b200.sh attach  CALL_ID                               # re-attach to a detached call (fc-...)
#
# TASKS     comma-separated task names or groups: relu_squared,rmsnorm  stage7  stage5,stage6  all
# BACKENDS  default triton (the CUDA C++ fixtures are validated only on sm_90)
# CASE_SET  default smoke; also acceptance, stratified, all, or one case ID
# TOOLS     default nsys; also ncu or nsys,ncu
#
# Env: MODAL_GPU (default B200), NSYS_PROFILE (default trace; e.g. stages or trace,graphs),
#      NCU_PRESET (default basic), ROW_TIMEOUT_S (default 900), RUN_ID (default: generated).
#
# Examples:
#   scripts/profile_b200.sh plan relu_squared,rmsnorm
#   scripts/profile_b200.sh launch relu_squared,rmsnorm triton smoke nsys
#   NSYS_PROFILE=stages scripts/profile_b200.sh launch stage7 triton acceptance nsys,ncu
#   scripts/profile_b200.sh fetch profile-20261114T180000Z-ab12cd34
#   scripts/profile_b200.sh kernels profile-20261114T180000Z-ab12cd34 5
#   scripts/profile_b200.sh query profile-20261114T180000Z-ab12cd34 tree --query top_level
set -euo pipefail
cd "$(dirname "$0")/.."

MODAL=.venv/bin/modal
VOLUME=agenticrl-cuda-artifacts
GPU="${MODAL_GPU:-B200}"

die() { echo "profile_b200.sh: $*" >&2; exit 1; }
usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

python_bin() {
    # query_nsys and the plan are Torch-free: the venv is used when present, else system python3.
    if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi
}

require_run_id() {
    [ -n "${1:-}" ] || die "RUN_ID required"
    [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || die "RUN_ID must be a plain directory name"
}

require_fetched() {
    # Checked in the parent shell: a die inside $(...) would only exit the subshell.
    [ -d "artifacts/runs/$1" ] || die "artifacts/runs/$1 not found; run: scripts/profile_b200.sh fetch $1"
}

captures() { find "artifacts/runs/$1" -name capture.sqlite | sort; }

plan_args() {
    [ -n "${1:-}" ] || die "TASKS required (for example relu_squared,rmsnorm or stage7)"
    PLAN=(--tasks "$1" --backends "${2:-triton}" --case-set "${3:-smoke}" --tools "${4:-nsys}"
          --nsys-profile "${NSYS_PROFILE:-trace}" --ncu-preset "${NCU_PRESET:-basic}"
          --timeout-s "${ROW_TIMEOUT_S:-900}")
}

cmd="${1:-help}"; shift || true
case "$cmd" in
    plan)
        plan_args "$@"
        PYTHONDONTWRITEBYTECODE=1 "$(python_bin)" -m scripts.profile_tasks "${PLAN[@]}" --dry-run
        ;;
    launch)
        plan_args "$@"
        [ -x "$MODAL" ] || die "$MODAL not found (install the project venv first)"
        # Fail on a bad request before any GPU is allocated.
        PYTHONDONTWRITEBYTECODE=1 "$(python_bin)" -m scripts.profile_tasks "${PLAN[@]}" --dry-run >/dev/null
        echo "profile_b200.sh: starting a PAID detached run on Modal $GPU" >&2
        MODAL_GPU="$GPU" PYTHONPATH="$PWD" "$MODAL" run --detach cloud/modal_profile.py \
            "${PLAN[@]}" ${RUN_ID:+--run-id "$RUN_ID"}
        ;;
    fetch)
        require_run_id "${1:-}"
        [ -x "$MODAL" ] || die "$MODAL not found"
        mkdir -p artifacts/runs
        [ ! -e "artifacts/runs/$1" ] || die "artifacts/runs/$1 already exists; move it away to fetch again"
        "$MODAL" volume get "$VOLUME" "/$1/" "artifacts/runs/$1"
        echo "saved to artifacts/runs/$1 (binary captures included: do not commit capture.* files)"
        ;;
    status)
        require_run_id "${1:-}"
        [ -x "$MODAL" ] || die "$MODAL not found"
        tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
        "$MODAL" volume get --force "$VOLUME" "/$1/profile/progress.jsonl" "$tmp" >/dev/null \
            || die "no progress yet for $1 (the run may still be building its image)"
        "$(python_bin)" - "$tmp" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    status = "skipped: " + r["skip"] if r["skip"] else (r.get("result") or {}).get("diag/status", r.get("driver_error"))
    print(f'{r["case_dir"]}  {status}')
PY
        ;;
    kernels)
        require_run_id "${1:-}"; require_fetched "$1"
        for sqlite in $(captures "$1"); do
            echo "== ${sqlite#artifacts/runs/$1/}"
            PYTHONDONTWRITEBYTECODE=1 "$(python_bin)" -m scripts.query_nsys kernels "$sqlite" --top "${2:-10}" \
                | "$(python_bin)" -c '
import json, sys
r = json.loads(sys.stdin.readlines()[-1])
print("   launches", r["launches"], " total", r["total_ns"], "ns  distinct kernels", r.get("distinct_kernels"))
for k in r["kernels"]:
    print("   %14d ns  x%-6d mean %12.1f ns  %s" % (k["total_ns"], k["count"], k["mean_ns"], k["name"]))'
        done
        ;;
    query)
        require_run_id "${1:-}"; run="$1"; shift
        [ "$#" -ge 1 ] || die "query needs a query_nsys command, e.g.: query $run nvtx"
        sub="$1"; shift
        require_fetched "$run"
        for sqlite in $(captures "$run"); do
            echo "== ${sqlite#artifacts/runs/$run/}"
            PYTHONDONTWRITEBYTECODE=1 "$(python_bin)" -m scripts.query_nsys "$sub" "$sqlite" "$@"
        done
        ;;
    attach)
        [ -n "${1:-}" ] || die "CALL_ID required (fc-...)"
        .venv/bin/python -m cloud.modal_job --call-id "$1"
        ;;
    help|-h|--help) usage 0 ;;
    *) usage 1 ;;
esac
