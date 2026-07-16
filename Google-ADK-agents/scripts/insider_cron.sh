#!/usr/bin/env bash
# Frequent poll of FMP insider-trading/latest -> important trades in TimescaleDB.
#
# The free feed is capped at the latest 100 filings (no pagination), and insider
# Form 4s file at hundreds/day, so poll OFTEN during US market hours to keep up.
# Idempotent upsert (trade_id) means overlap is harmless. ~1 FMP request/run.
#
# Suggested crontab — every 30 min, 14:00-22:00 BST, Mon-Fri (covers the US
# session + after-close filing window):
#   */30 14-22 * * 1-5  <path>/scripts/insider_cron.sh
#
# Logs to $INSIDER_SEED_LOG (default /tmp/insider_seed.log). INSIDER_MIN_VALUE
# overrides the $50k materiality threshold.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Repo often lives on /mnt/c; keep uv's env on the Linux filesystem (see
# seed_cron.sh) to avoid the Windows-.venv repair I/O error.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/google-adk-agents}"

LOG="${INSIDER_SEED_LOG:-/tmp/insider_seed.log}"

{
  echo "=== $(date -u +%FT%TZ) insider poll start ==="
  uv run --extra timescale python seed_insider_trades.py
  echo "=== $(date -u +%FT%TZ) insider poll done ==="
} >> "$LOG" 2>&1
