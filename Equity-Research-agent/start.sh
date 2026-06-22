#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# Install version-specific venv package
sudo apt install -y python3.12-venv

# Recreate venv if missing or broken
if [ ! -d ".venv" ] || ! .venv/bin/python3 -c "" &>/dev/null; then
  rm -rf .venv
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

# Free port 8000 if already in use
PORT=8000
PID=$(lsof -t -i:$PORT 2>/dev/null || true)
if [ -n "$PID" ]; then
  echo "Port $PORT in use (PID $PID) — killing..."
  kill "$PID" && sleep 1
fi

echo ""
echo "Equity Research API → http://localhost:$PORT"
echo "Docs            → http://localhost:$PORT/docs"
echo ""
uvicorn api:app --reload --port $PORT
