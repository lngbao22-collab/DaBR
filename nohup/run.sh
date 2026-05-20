#!/bin/bash

# run.sh - Start DaBR training/prediction in the background
# Usage: ./run.sh [--dataset DATASET] [--config CONFIG_FILE] [--mode MODE] [--checkpoint_path PATH]
# Examples:
#   ./run.sh --dataset WN18RR --config ./config/WN18RR_test.json --mode train
#   ./run.sh --dataset FB15K237 --config ./config/FB15K237.json --mode train
#   ./run.sh --dataset WN18RR --config ./config/WN18RR_test.json --mode predict --checkpoint_path ./logs/WN18RR/checkpoints/.../DaBR.ckpt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PID_FILE="$SCRIPT_DIR/.pid"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Error: Process already running with PID $OLD_PID"
        echo "Use ./stop.sh to stop it first, or ./check.sh to see status"
        exit 1
    fi
fi

# Run in background with nohup
cd "$PROJECT_DIR"
echo "Starting DaBR process with arguments: $@"
nohup "$VENV_PYTHON" main.py "$@" > "$SCRIPT_DIR/nohup.out" 2>&1 &
NEW_PID=$!

# Save PID to file
echo "$NEW_PID" > "$PID_FILE"

echo "Process started with PID $NEW_PID"
echo "Output is being logged to: $SCRIPT_DIR/nohup.out"
echo "Use './check.sh' to check status or './stop.sh' to stop the process"
