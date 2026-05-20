#!/bin/bash

# stop.sh - Stop the background DaBR process
# Usage: ./stop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: No process information found. Process may not have been started with run.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")

# Check if process is still running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process with PID $PID is not running"
    rm -f "$PID_FILE"
    exit 0
fi

# Terminate the process gracefully first
echo "Stopping process with PID $PID..."
kill -TERM "$PID"

# Wait a bit for graceful shutdown
sleep 2

# Check if still running, force kill if needed
if kill -0 "$PID" 2>/dev/null; then
    echo "Process still running, forcing kill..."
    kill -KILL "$PID"
fi

# Clean up PID file
rm -f "$PID_FILE"
echo "Process stopped"
