#!/bin/bash
# NFL Betting API Server - Double-click to start
# This opens Terminal and starts the server with auto-setup

cd "$(dirname "$0")"
echo "Starting NFL Betting API Server..."
echo ""
echo "The server will:"
echo "  1. Initialize database"
echo "  2. Start API on http://localhost:8000"
echo "  3. Auto-populate with current week data"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 nfl_manager.py start

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "Press any key to close..."
    read -n 1
fi
