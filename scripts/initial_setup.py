#!/usr/bin/env python3
"""Initial setup script for NFL betting system.

Run this once to:
1. Initialize SQLite database
2. Download all nflverse data for 2025 season (~200MB)
3. Fetch current odds (requires ODDS_API_KEY)
4. Fetch current injuries

Usage:
    python scripts/initial_setup.py

    # Skip odds if no API key
    python scripts/initial_setup.py --skip-odds

    # Force re-download everything
    python scripts/initial_setup.py --force
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Initial setup for NFL betting system')
    parser.add_argument('--season', type=int, default=2025,
                       help='NFL season year (default: 2025)')
    parser.add_argument('--skip-odds', action='store_true',
                       help='Skip odds fetch (if no ODDS_API_KEY)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download all data')
    args = parser.parse_args()

    print("=" * 60)
    print(f"NFL Betting System - Initial Setup")
    print(f"Season: {args.season}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n[1/5] Initializing SQLite database...")
    try:
        from backend.database.local_db import init_database, DB_PATH
        init_database()
        print(f"  ✓ Database initialized: {DB_PATH}")
    except Exception as e:
        print(f"  ✗ Database init failed: {e}")
        return 1

    # Step 2: Create directories
    print("\n[2/5] Creating directories...")
    dirs = [
        PROJECT_ROOT / "inputs",
        PROJECT_ROOT / "inputs" / "injuries",
        PROJECT_ROOT / "inputs" / "odds",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "data",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")

    # Step 3: Download nflverse data
    print("\n[3/5] Downloading nflverse data (this may take a few minutes)...")
    try:
        from backend.ingestion.fetch_nflverse import fetch_nflverse
        from backend.ingestion.fetch_nflverse_schedules import fetch_schedule

        out_dir = PROJECT_ROOT / "inputs"

        # Fetch main nflverse data
        fetch_nflverse(args.season, out_dir, force=args.force)

        # Fetch schedule
        fetch_schedule(args.season, out_dir, force=args.force)

        print("  ✓ nflverse data downloaded")
    except Exception as e:
        print(f"  ✗ nflverse download failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: Fetch injuries
    print("\n[4/5] Fetching current injuries...")
    try:
        from backend.ingestion.fetch_injuries import fetch_injuries

        injuries_dir = PROJECT_ROOT / "inputs" / "injuries"
        fetch_injuries(injuries_dir)
        print("  ✓ Injuries fetched")
    except Exception as e:
        print(f"  ✗ Injuries fetch failed: {e}")

    # Step 5: Fetch odds (if API key available)
    if args.skip_odds:
        print("\n[5/5] Skipping odds fetch (--skip-odds)")
    else:
        print("\n[5/5] Fetching current odds...")
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            print("  ⚠ ODDS_API_KEY not set - skipping odds fetch")
            print("  To fetch odds later:")
            print("    export ODDS_API_KEY=your_key_here")
            print("    curl -X POST http://localhost:8000/fetch/odds")
        else:
            try:
                from backend.ingestion.fetch_odds import fetch_all_props

                odds_dir = PROJECT_ROOT / "inputs" / "odds"
                fetch_all_props(api_key, output_dir=odds_dir)

                # Also insert into database
                from backend.database.local_db import OddsRepository
                # Note: This would need the parsed data, skipping for now
                print("  ✓ Odds fetched and saved to files")
                print("  Note: Use /fetch/odds API endpoint to also insert into database")
            except Exception as e:
                print(f"  ✗ Odds fetch failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)

    # Check what was downloaded
    inputs_dir = PROJECT_ROOT / "inputs"
    files = list(inputs_dir.glob("*.csv")) + list(inputs_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)

    print(f"\nData summary:")
    print(f"  Files in inputs/: {len(files)}")
    print(f"  Total size: {total_size:.1f} MB")

    # Check database
    if DB_PATH.exists():
        db_size = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"  Database size: {db_size:.1f} MB")

    print(f"\nNext steps:")
    print(f"  1. Start API server: python api_server.py")
    print(f"  2. Fetch odds: curl -X POST http://localhost:8000/fetch/odds")
    print(f"  3. Generate picks: curl http://localhost:8000/picks/passing")

    if not os.getenv("ODDS_API_KEY"):
        print(f"\n  Note: Set ODDS_API_KEY for real sportsbook lines")
        print(f"    export ODDS_API_KEY=your_key_here")

    return 0


if __name__ == "__main__":
    sys.exit(main())
