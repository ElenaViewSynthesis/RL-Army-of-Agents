#!/usr/bin/env bash
# Daily TimescaleDB seed — intended to run from cron.
#
# Feeds fresh raw points into the commodity_prices hypertable across the 11-code
# energy complex. The commodity_prices_daily continuous aggregate then refreshes
# itself hourly (its own Timescale policy), so this only needs to keep raw data
# flowing. Idempotent: overlapping day windows upsert, so a missed run self-heals.
#
# Budget: 11 OilPrice requests/run (5.5% of the 200/day cap) + 5 FMP snapshots.
# Set SEED_NO_FMP=1 to drop the FMP snapshot (OilPrice-only, 11 requests).
#
# Install (WSL, from any shell):
#   sudo service cron start            # WSL2 doesn't auto-start cron
#   ( crontab -l 2>/dev/null; echo "15 6 * * * $(pwd)/scripts/seed_cron.sh" ) | crontab -
#   crontab -l                         # verify
# Logs to $TIGER_SEED_LOG (default /tmp/tiger_seed.log).

set -euo pipefail

# Resolve the project dir (Google-ADK-agents/) regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# The repo often lives on /mnt/c (Windows filesystem); a Windows-created .venv
# there cannot be repaired by Linux uv (fails with I/O error os 5). Put uv's
# project environment on the Linux filesystem — avoids that and is much faster.
# Pre-set UV_PROJECT_ENVIRONMENT to override.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/google-adk-agents}"

LOG="${TIGER_SEED_LOG:-/tmp/tiger_seed.log}"
ARGS=(--days 4)
[ "${SEED_NO_FMP:-0}" = "1" ] && ARGS+=(--no-fmp)

{
  echo "=== $(date -u +%FT%TZ) seed start (${ARGS[*]}) ==="
  uv run --extra timescale python seed_timescale_prices.py "${ARGS[@]}"
  echo "=== $(date -u +%FT%TZ) seed done ==="
} >> "$LOG" 2>&1
