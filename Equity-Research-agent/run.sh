#!/bin/bash

cd "$(dirname "$0")"

set -a
source .env
set +a

TICKER=${1:-AAPL}
SAVE=${2:---save}

node agent.js "$TICKER" $SAVE
