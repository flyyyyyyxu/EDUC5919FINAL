#!/bin/bash
# Run the simulation and log all output to a timestamped file.
# Usage: bash run_sim.sh

LOG_DIR="../../environment/frontend_server/storage/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/sim_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: $LOGFILE"
python reverie.py 2>&1 | tee "$LOGFILE"
