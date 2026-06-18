#!/bin/bash

cd "C:/Users/proxi/Documents/codex3/RL-Army-of-Agents/Equity-Research-agent"

set -a
source .env
set +a

TICKER=${1:-AAPL}
SAVE=${2:---save}

node agent.js "$TICKER" $SAVE
