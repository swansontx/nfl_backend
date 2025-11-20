#!/usr/bin/env python3
"""Start the NFL betting API server.

Run with: python start_server.py
Or: ./start_server.py
Or: python start_server.py --auto-update  (populate database on startup)

The API server will start on http://localhost:8000
"""

import argparse
import subprocess
import sys
import asyncio
import requests
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent


def auto_update_database(season: int = 2025, week: int = 12):
    """Auto-update database with fresh data after server starts."""
    print("\nAuto-updating database...")
    print("=" * 50)

    # Wait for server to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Warning: Server not ready for auto-update")
        return

    # Call populate/all endpoint
    try:
        print(f"Populating database for {season} season, week {week}...")

        # Populate all data including odds
        response = requests.post(
            "http://localhost:8000/populate/all",
            params={
                "season": season,
                "week": week,
                "fetch_first": False,
                "include_odds": True
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print("\nDatabase populated successfully!")
            print("-" * 50)
            for key, value in result.get('results', {}).items():
                print(f"  {key}: {value}")
            print("-" * 50)
            print("\nNext steps:")
            for step in result.get('next_steps', []):
                print(f"  - {step}")
        else:
            print(f"Warning: Population failed with status {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Warning: Auto-update failed: {e}")

    print("=" * 50)
    print("\nServer is running! Use Ctrl+C to stop.\n")


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="Start NFL Betting API Server")
    parser.add_argument(
        "--auto-update", "-a",
        action="store_true",
        help="Auto-populate database on startup"
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        default=2025,
        help="NFL season (default: 2025)"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        default=12,
        help="Current NFL week (default: 12)"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (for production)"
    )

    args = parser.parse_args()

    print("Starting NFL Betting API Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 50)
    print("\nPress Ctrl+C to stop the server\n")

    # Initialize database
    from backend.database.local_db import init_database
    db_path = init_database()
    print(f"Database: {db_path}")

    if args.auto_update:
        print(f"\nAuto-update enabled for {args.season} season, week {args.week}")
        print("Database will be populated after server starts...")

        # Start uvicorn in background, then auto-update
        import threading

        def run_server():
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api_server:app",
                "--host", "0.0.0.0",
                "--port", "8000"
            ]
            if not args.no_reload:
                cmd.append("--reload")

            subprocess.run(cmd, cwd=PROJECT_ROOT)

        # Start server in thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait a moment then auto-update
        time.sleep(3)
        auto_update_database(args.season, args.week)

        # Keep main thread alive
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            sys.exit(0)
    else:
        # Normal startup
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        if not args.no_reload:
            cmd.append("--reload")

        subprocess.run(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
