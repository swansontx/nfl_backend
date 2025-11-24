#!/usr/bin/env bash
# NFL Betting API Server - Double-click to start
# This opens Terminal and starts the server with auto-setup
# Supports automation tools (e.g., Goose extensions) via GOOSE_EXTENSIONS=1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Resolve python command (prefer python3)
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is not installed. Please install Python 3." >&2
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

# Optional automation mode: skip pauses/log to file when GOOSE_EXTENSIONS=1
if [[ "${GOOSE_EXTENSIONS:-0}" == "1" ]]; then
  LOG_FILE="$PROJECT_ROOT/goose_start.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

# Setup virtual environment if present or needed
if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
  PYTHON_BIN="python"
elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
  source "$PROJECT_ROOT/.venv/bin/activate"
  PYTHON_BIN="python"
  echo "Installing dependencies..."
  "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

# Ensure project dependencies are installed when a venv isn't used
if ! "$PYTHON_BIN" - <<'PY' 2>/dev/null; then
import requests  # simple import check for common deps
PY
  echo "Installing dependencies..."
  "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

clear
cat <<'MSG'
Starting NFL Betting API Server...

The server will:
  1. Initialize database
  2. Start API on http://localhost:8000
  3. Auto-populate with current week data

Press Ctrl+C to stop the server
MSG

echo ""
# Start the manager in auto mode
if ! "$PYTHON_BIN" "$PROJECT_ROOT/nfl_manager.py" start; then
  STATUS=$?
  echo "\nServer failed to start (exit code $STATUS)."
  if [[ "${GOOSE_EXTENSIONS:-0}" != "1" ]]; then
    read -n 1 -s -r -p "Press any key to close..."
  fi
  exit "$STATUS"
fi

# Keep terminal open when launched interactively (not Goose automation)
if [[ "${GOOSE_EXTENSIONS:-0}" != "1" ]]; then
  echo "\nServer exited. Press any key to close..."
  read -n 1 -s -r
fi
