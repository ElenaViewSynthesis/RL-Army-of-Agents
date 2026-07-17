#!/usr/bin/env bash
# Periodic snapshot of OilPrice marine-ports into the TimescaleDB columnstore.
#
# Reference data that changes rarely, so run infrequently — each run stamps a new
# snapshot_ts (a point-in-time snapshot); the columnstore compresses the history.
# ~1 OilPrice request/run. Idempotent upsert on (code, snapshot_ts).
#
# Suggested crontab — weekly, Monday 03:00 (WSL local time):
#   0 3 * * 1  <path>/scripts/marine_cron.sh
#
# Logs to $MARINE_SEED_LOG (default /tmp/marine_seed.log).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Repo often lives on /mnt/c; keep uv's env on the Linux filesystem (see
# seed_cron.sh) to avoid the Windows-.venv repair I/O error.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/google-adk-agents}"

LOG="${MARINE_SEED_LOG:-/tmp/marine_seed.log}"

{
  echo "=== $(date -u +%FT%TZ) marine snapshot start ==="
  uv run --extra timescale python seed_marine_ports.py
  echo "=== $(date -u +%FT%TZ) marine snapshot done ==="
} >> "$LOG" 2>&1
