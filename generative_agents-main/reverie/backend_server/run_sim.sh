#!/bin/bash
# Run the simulation and log all output to a timestamped file.
# Usage: bash run_sim.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
LOG_DIR="../../environment/frontend_server/storage/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/sim_$(date +%Y%m%d_%H%M%S).log"

if [ ! -x "$VENV_PY" ]; then
  echo "Missing virtualenv python: $VENV_PY"
  exit 1
fi

if ! "$VENV_PY" "$SCRIPT_DIR/preflight.py"; then
  exit 1
fi

echo "Logging to: $LOGFILE"
"$VENV_PY" reverie.py 2>&1 | tee "$LOGFILE"
