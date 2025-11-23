#!/usr/bin/env python3
"""NFL Betting System Manager - Single entry point for everything.

This script manages all aspects of the NFL betting system:
- Automatic season/week detection
- Data download and updates
- Database initialization and population
- API server startup
- MCP server for Claude Desktop

Usage:
    python nfl_manager.py              # Interactive menu
    python nfl_manager.py start        # Start API server with auto-setup
    python nfl_manager.py setup        # Download data and setup database
    python nfl_manager.py update       # Update all data sources
    python nfl_manager.py status       # Check system status
    python nfl_manager.py mcp          # Start MCP server (for Claude Desktop)
"""

import argparse
import asyncio
import subprocess
import sys
import time
import os
import json
from pathlib import Path
from datetime import datetime

import requests

# Project root
PROJECT_ROOT = Path(__file__).parent

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))

from backend.nfl_calendar import get_current_season_and_week, is_regular_season
from backend.config import settings
from backend.database.local_db import init_database


class NFLManager:
    """Manager for NFL betting system."""

    def __init__(self):
        self.season, self.week = get_current_season_and_week()
        self.api_url = f"http://localhost:{settings.api_port}"
        self.db_path = PROJECT_ROOT / "data" / "nfl_betting.db"
        self.inputs_dir = PROJECT_ROOT / "inputs"

    def print_header(self):
        """Print system header with current season info."""
        print("\n" + "=" * 60)
        print("  NFL Betting System Manager")
        print("=" * 60)
        print(f"  Season: {self.season}  |  Week: {self.week}")
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60 + "\n")

    def check_api_server(self) -> bool:
        """Check if API server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def wait_for_api(self, timeout: int = 30) -> bool:
        """Wait for API server to be ready."""
        print("Waiting for API server...", end="", flush=True)
        start = time.time()
        while time.time() - start < timeout:
            if self.check_api_server():
                print(" Ready!")
                return True
            print(".", end="", flush=True)
            time.sleep(1)
        print(" Timeout!")
        return False

    def status(self):
        """Check system status."""
        self.print_header()
        print("System Status:")
        print("-" * 40)

        # Check database
        if self.db_path.exists():
            size_mb = self.db_path.stat().st_size / (1024 * 1024)
            print(f"  Database: {size_mb:.1f} MB")
        else:
            print("  Database: Not initialized")

        # Check inputs directory
        if self.inputs_dir.exists():
            csv_files = list(self.inputs_dir.glob("*.csv"))
            parquet_files = list(self.inputs_dir.glob("*.parquet"))
            print(f"  Input files: {len(csv_files)} CSV, {len(parquet_files)} Parquet")
        else:
            print("  Input files: None")

        # Check API server
        if self.check_api_server():
            print("  API Server: Running")
            # Get data freshness
            try:
                response = requests.get(f"{self.api_url}/refresh/check", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print("\n  Data Freshness:")
                    for source, info in data.get("sources", {}).items():
                        status = "Fresh" if info.get("is_fresh") else "Stale"
                        print(f"    - {source}: {status}")
            except:
                pass
        else:
            print("  API Server: Not running")

        # Check environment
        print("\n  Environment:")
        if settings.odds_api_key:
            print("    - Odds API: Configured")
        else:
            print("    - Odds API: Not configured (set ODDS_API_KEY)")

        print("-" * 40)

    def setup(self):
        """Initial setup - download data and initialize database."""
        self.print_header()
        print("Running initial setup...")
        print("-" * 40)

        # Initialize database
        print("\n1. Initializing database...")
        db_path = init_database()
        print(f"   Database: {db_path}")

        # Check for input data
        print("\n2. Checking input data...")
        self.inputs_dir.mkdir(exist_ok=True)

        # Look for existing data files
        stats_files = list(self.inputs_dir.glob("player_stats*.csv"))
        schedule_files = list(self.inputs_dir.glob("schedule*.csv"))

        if stats_files:
            print(f"   Found {len(stats_files)} stats files")
        else:
            print("   No stats files found - will need to fetch from nflverse")

        if schedule_files:
            print(f"   Found {len(schedule_files)} schedule files")
        else:
            print("   No schedule files found - will need to fetch from nflverse")

        print("\n3. Setup complete!")
        print("\nNext steps:")
        print("  - Run 'python nfl_manager.py start' to start the API server")
        print("  - The server will auto-populate the database on first start")
        print("-" * 40)

    def start(self, auto_update: bool = True, background: bool = False):
        """Start the API server with auto-setup."""
        self.print_header()

        # Initialize database first
        print("Initializing database...")
        init_database()

        if background:
            print("Starting API server in background...")
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api_server:app",
                "--host", "0.0.0.0",
                "--port", str(settings.api_port)
            ]
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Server started (PID: {process.pid})")

            if auto_update and self.wait_for_api():
                self._populate_database()
        else:
            print(f"Starting API server on port {settings.api_port}...")
            print(f"API docs: {self.api_url}/docs")
            print("\nPress Ctrl+C to stop\n")

            if auto_update:
                # Start in background thread, then populate
                import threading

                def run_server():
                    cmd = [
                        sys.executable, "-m", "uvicorn",
                        "api_server:app",
                        "--host", "0.0.0.0",
                        "--port", str(settings.api_port),
                        "--reload"
                    ]
                    subprocess.run(cmd, cwd=PROJECT_ROOT)

                server_thread = threading.Thread(target=run_server, daemon=True)
                server_thread.start()

                if self.wait_for_api():
                    self._populate_database()
                    print("\nServer is running! Use Ctrl+C to stop.\n")

                try:
                    server_thread.join()
                except KeyboardInterrupt:
                    print("\nShutting down...")
            else:
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "api_server:app",
                    "--host", "0.0.0.0",
                    "--port", str(settings.api_port),
                    "--reload"
                ]
                subprocess.run(cmd, cwd=PROJECT_ROOT)

    def _populate_database(self):
        """Populate database with current season data."""
        print(f"\nPopulating database for {self.season} season, week {self.week}...")
        print("-" * 40)

        try:
            response = requests.post(
                f"{self.api_url}/populate/all",
                params={
                    "season": self.season,
                    "week": self.week,
                    "fetch_first": False,
                    "include_odds": True
                },
                timeout=180
            )

            if response.status_code == 200:
                result = response.json()
                print("\nDatabase populated successfully!")
                for key, value in result.get("results", {}).items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nWarning: Population failed ({response.status_code})")
                print(response.text[:200])
        except Exception as e:
            print(f"\nWarning: Population failed: {e}")

        print("-" * 40)

    def update(self, force: bool = False):
        """Update all data sources."""
        self.print_header()

        if not self.check_api_server():
            print("Error: API server not running")
            print("Start it first: python nfl_manager.py start")
            return

        print(f"Updating data for {self.season} season, week {self.week}...")
        print("-" * 40)

        try:
            # Auto-refresh stale data
            response = requests.post(
                f"{self.api_url}/refresh/auto",
                params={
                    "week": self.week,
                    "year": self.season,
                    "force": force
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                print("\nData updated successfully!")
                for key, value in result.get("refreshed", {}).items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nWarning: Update failed ({response.status_code})")
        except Exception as e:
            print(f"\nError: {e}")

        print("-" * 40)

    def fetch_nflverse(self):
        """Fetch fresh nflverse data."""
        self.print_header()

        if not self.check_api_server():
            print("Error: API server not running")
            return

        print(f"Fetching nflverse data for {self.season}...")
        print("This may take several minutes...")
        print("-" * 40)

        try:
            response = requests.post(
                f"{self.api_url}/fetch/nflverse",
                params={"year": self.season, "include_all": True},
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                print("\nnflverse data fetched successfully!")
                if "files_saved" in result:
                    for f in result["files_saved"]:
                        print(f"  - {f}")
            else:
                print(f"\nWarning: Fetch failed ({response.status_code})")
        except Exception as e:
            print(f"\nError: {e}")

        print("-" * 40)

    def mcp(self):
        """Start the MCP server for Claude Desktop."""
        # MCP server runs via stdio, just exec it
        import asyncio
        from mcp_server import main
        asyncio.run(main())

    def interactive_menu(self):
        """Show interactive menu."""
        while True:
            self.print_header()
            print("Options:")
            print("  1. Start API server (with auto-setup)")
            print("  2. Check system status")
            print("  3. Update all data")
            print("  4. Fetch fresh nflverse data")
            print("  5. Initial setup")
            print("  6. Start MCP server (for Claude Desktop)")
            print("  0. Exit")
            print("-" * 40)

            try:
                choice = input("Select option: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if choice == "1":
                self.start()
            elif choice == "2":
                self.status()
                input("\nPress Enter to continue...")
            elif choice == "3":
                self.update()
                input("\nPress Enter to continue...")
            elif choice == "4":
                self.fetch_nflverse()
                input("\nPress Enter to continue...")
            elif choice == "5":
                self.setup()
                input("\nPress Enter to continue...")
            elif choice == "6":
                self.mcp()
            elif choice == "0":
                print("\nGoodbye!")
                break
            else:
                print("Invalid option")
                time.sleep(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Betting System Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start     Start API server with auto-setup
  setup     Initial setup (database, check data)
  update    Update all data sources
  fetch     Fetch fresh nflverse data
  status    Check system status
  mcp       Start MCP server for Claude Desktop

Examples:
  python nfl_manager.py              # Interactive menu
  python nfl_manager.py start        # Quick start
  python nfl_manager.py status       # Check status
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["start", "setup", "update", "fetch", "status", "mcp"],
        help="Command to run"
    )
    parser.add_argument(
        "--no-auto-update", "-n",
        action="store_true",
        help="Don't auto-update database on start"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force update even if data is fresh"
    )
    parser.add_argument(
        "--background", "-b",
        action="store_true",
        help="Run server in background"
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        help="Override season year"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        help="Override week number"
    )

    args = parser.parse_args()

    manager = NFLManager()

    # Override season/week if specified
    if args.season:
        manager.season = args.season
    if args.week:
        manager.week = args.week

    # Run command
    if args.command is None:
        manager.interactive_menu()
    elif args.command == "start":
        manager.start(
            auto_update=not args.no_auto_update,
            background=args.background
        )
    elif args.command == "setup":
        manager.setup()
    elif args.command == "update":
        manager.update(force=args.force)
    elif args.command == "fetch":
        manager.fetch_nflverse()
    elif args.command == "status":
        manager.status()
    elif args.command == "mcp":
        manager.mcp()


if __name__ == "__main__":
    main()
