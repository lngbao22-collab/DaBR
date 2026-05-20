#!/bin/bash

# check.sh - Check status of the background DaBR process
# Usage: ./check.sh [tail_lines]
# Examples:
#   ./check.sh           # Show status and last 20 lines of output
#   ./check.sh 50        # Show status and last 50 lines of output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pid"
LOG_FILE="$SCRIPT_DIR/nohup.out"
TAIL_LINES=${1:-20}

echo "=== DaBR Background Process Status ==="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no process file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Status: RUNNING"
    echo "PID: $PID"
    
    # Get process info
    if command -v ps &> /dev/null; then
        echo "Process Info:"
        ps -p "$PID" -o pid,ppid,cmd,etime,lstart | tail -1 | awk '{for(i=5;i<=NF;i++) printf "%s ", $i; print ""}'
    fi
    
    # Show last lines of output
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== Last $TAIL_LINES lines of output (from $LOG_FILE): ==="
        tail -n "$TAIL_LINES" "$LOG_FILE"
    fi
else
    echo "Status: NOT RUNNING"
    echo "PID $PID is no longer active"
    rm -f "$PID_FILE"
fi

echo ""
echo "=== Commands ==="
echo "  ./run.sh [args]   - Start a new process"
echo "  ./stop.sh         - Stop the running process"
echo "  ./check.sh [N]    - Check status (show last N lines, default 20)"
