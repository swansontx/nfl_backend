#!/usr/bin/env python3
"""Start the NFL betting API server.

Run with: python start_server.py
Or: ./start_server.py

The API server will start on http://localhost:8000
"""

import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

def main():
    """Start the API server."""
    print("Starting NFL Betting API Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 50)
    print("\nPress Ctrl+C to stop the server\n")

    # Initialize database
    from backend.database.local_db import init_database
    db_path = init_database()
    print(f"Database: {db_path}\n")

    # Start uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
