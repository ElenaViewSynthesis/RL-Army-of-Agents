#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── ensure python3-venv is available ─────────────────────────────────────────
if ! python3 -m venv --help &>/dev/null; then
  echo "Installing python3-venv..."
  sudo apt install python3-venv python3-pip -y
fi

# ── create venv if it doesn't exist ──────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# ── activate ──────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── install / sync dependencies ───────────────────────────────────────────────
echo "Installing dependencies..."
pip install -q -r requirements.txt

# ── start server ──────────────────────────────────────────────────────────────
echo ""
echo "Starting Equity Research API on http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo ""
uvicorn api:app --reload --port 8000
