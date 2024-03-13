#!/bin/bash

# Run the first command in the background and capture its output (which is the process ID)
python3 -m src.main &

# Capture the PID of the background process
pid=$!

# Check if the PID is valid
if [[ -n "$pid" ]]; then
    echo "Process ID: $pid"

    # Run the second command with the captured PID
    python3 measure_main.py "$pid" measurements
else
    echo "Failed to retrieve process ID."
    exit 1
fi
