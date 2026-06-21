#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── ensure python3-venv + ensurepip are available ────────────────────────────
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ! python3 -m ensurepip --version &>/dev/null; then
  echo "Installing python${PY_VER}-venv..."
  sudo apt install -y "python${PY_VER}-venv"
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
