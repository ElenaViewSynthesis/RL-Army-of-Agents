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

echo ""
echo "Equity Research API → http://localhost:8000"
echo "Docs            → http://localhost:8000/docs"
echo ""
uvicorn api:app --reload --port 8000
