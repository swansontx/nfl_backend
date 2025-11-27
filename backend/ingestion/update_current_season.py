"""Weekly Update Tool for Current Season Data.

Fetches the latest play-by-play data and updates player statistics
for the ongoing season (2025). Run this weekly or after each game week
to keep the system current.

Usage:
    python -m backend.ingestion.update_current_season
    python -m backend.ingestion.update_current_season --season 2025 --verbose
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.backtesting.data_collector import HistoricalDataCollector


def update_current_season(season: int = 2025, verbose: bool = False):
    """Update current season data from latest play-by-play.

    Args:
        season: Season year to update (default: 2025)
        verbose: Print detailed progress

    Returns:
        dict: Summary of updated data
    """
    print(f"{'='*70}")
    print(f"NFL CURRENT SEASON DATA UPDATER")
    print(f"{'='*70}")
    print(f"Season: {season}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    collector = HistoricalDataCollector()

    # Collect current season data
    print(f"Fetching latest data for {season} season...\n")

    try:
        data = collector.collect_season_data(season, source='nfl_data_py')

        # Summary
        print(f"\n{'='*70}")
        print("UPDATE SUMMARY")
        print(f"{'='*70}")

        summary = {
            'season': season,
            'timestamp': datetime.now().isoformat(),
            'games': len(data.get('games', [])),
            'player_stats': len(data.get('player_stats', [])),
            'injuries': len(data.get('injuries', [])),
            'rosters': len(data.get('rosters', [])) if 'rosters' in data else 0
        }

        print(f"✓ Games: {summary['games']}")
        print(f"✓ Player-week records: {summary['player_stats']}")
        print(f"✓ Injuries: {summary['injuries']}")
        print(f"✓ Rosters: {summary['rosters']}")

        # Show weeks available
        if len(data.get('player_stats', [])) > 0:
            weeks = sorted(data['player_stats']['week'].unique())
            print(f"\nWeeks available: {weeks}")
            print(f"Latest week: {max(weeks)}")

            if verbose:
                # Show position breakdown
                positions = data['player_stats']['position'].value_counts()
                print(f"\nPosition breakdown:")
                for pos, count in positions.items():
                    print(f"  {pos}: {count}")

        print(f"\n{'='*70}")
        print(f"✓ Update complete! Data saved to inputs/historical/")
        print(f"{'='*70}\n")

        # Show next steps
        print("Next steps:")
        print("  1. Re-run model training with updated data")
        print("  2. Generate fresh predictions for current week")
        print("  3. Update backtesting validation (optional)")
        print()

        return summary

    except Exception as e:
        print(f"\n✗ Error updating season data: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Update current season data from latest play-by-play'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='Season year to update (default: 2025)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify data without updating'
    )

    args = parser.parse_args()

    if args.verify:
        # Just check what data is available
        collector = HistoricalDataCollector()
        avail = collector.verify_data_availability([args.season])

        print(f"\nData availability for {args.season}:")
        for season, data in avail.items():
            print(f"  Games: {data['games_count']}")
            print(f"  Player stats: {data['stats_count']}")
            print(f"  Injuries: {data['injuries_count']}")
    else:
        # Update the data
        result = update_current_season(
            season=args.season,
            verbose=args.verbose
        )

        if result is None:
            sys.exit(1)


if __name__ == '__main__':
    main()
