#!/bin/bash
# NFL Betting System Manager - Double-click to run
# This opens Terminal and starts the interactive manager

cd "$(dirname "$0")"
echo "Starting NFL Betting System Manager..."
echo ""
python3 nfl_manager.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "Press any key to close..."
    read -n 1
fi
